// https://github.com/gunnarmorling/1brc/discussions/46
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

// Capacity of our hashmap
// Since we use linear probing this needs to be at least twice as big
// as the # of distinct strings in our dataset
// Also must be power of 2 so we can use bit-and instead of modulo
#define HCAP (4096 * 4)
#define MAX_DISTINCT_GROUPS 16384
#define MAX_GROUPBY_KEY_LENGTH 100
#define NTHREADS 128

// branchless min/max (on some machines at least)
#define min(a, b) (a ^ ((b ^ a) & -(b < a)));
#define max(a, b) (a ^ ((a ^ b) & -(a < b)));

// parses a floating point number as an integer
// this is only possible because we know our data file has only a single decimal
static inline const char *parse_number(int *dest, const char *s) {

  // parse sign
  int mod;
  if (*s == '-') {
    mod = -1;
    s++;
  } else {
    mod = 1;
  }

  if (s[1] == '.') {
    *dest = ((s[0] * 10) + s[2] - ('0' * 11)) * mod;
    return s + 4;
  }

  *dest = (s[0] * 100 + s[1] * 10 + s[3] - '0' * 111) * mod;
  return s + 5;
}

// hash returns a simple (but fast) hash for the first n bytes of data
static unsigned int hash(const unsigned char *data, int n) {
  unsigned int hash = 0;

  for (int i = 0; i < n; i++) {
    hash = (hash * 31) + data[i];
  }

  return hash;
}

struct Group {
  unsigned int count;
  long sum;
  int min;
  int max;
  char *label;
};

struct Result {
  int map[HCAP];
  int n;
  char labels[MAX_DISTINCT_GROUPS][MAX_GROUPBY_KEY_LENGTH];
  struct Group groups[MAX_DISTINCT_GROUPS];
};

struct Chunk {
  size_t start;
  size_t end;
  const char *data;
};

// qsort callback
static int cmp(const void *ptr_a, const void *ptr_b) {
  return strcmp(((struct Group *)ptr_a)->label, ((struct Group *)ptr_b)->label);
}

static inline unsigned int
hash_probe(int map[HCAP],
           char groups[MAX_DISTINCT_GROUPS][MAX_GROUPBY_KEY_LENGTH],
           const char *start, int len) {
  // probe map until free spot or match
  unsigned int h = hash((unsigned char *)start, len) & (HCAP - 1);
  while (map[h] >= 0 && memcmp(groups[map[h]], start, (size_t)len) != 0) {
    h = (h + 1) & (HCAP - 1);
  }

  return h;
}

static void *process_chunk(void *ptr) {
  struct Chunk *ch = (struct Chunk *)ptr;

  // skip start forward until SOF or after next newline
  if (ch->start > 0) {
    while (ch->data[ch->start - 1] != '\n') {
      ch->start++;
    }
  }

  while (ch->data[ch->end] != 0x0 && ch->data[ch->end - 1] != '\n') {
    ch->end++;
  }

  struct Result *result = malloc(sizeof(*result));
  if (!result) {
    perror("malloc error");
    exit(EXIT_FAILURE);
  }
  result->n = 0;
  memset(result->labels, 0,
         MAX_DISTINCT_GROUPS * MAX_GROUPBY_KEY_LENGTH * sizeof(char));
  memset(result->map, -1, HCAP * sizeof(int));

  const char *s = &ch->data[ch->start];
  const char *end = &ch->data[ch->end];
  const char *linestart;
  unsigned int h;
  int temperature;
  int len;
  int c;

  while (s != end) {
    linestart = s;

    // hash everything up to ';'
    // assumption: key is at least 1 char
    len = 1;
    h = (unsigned char)s[0];
    while (s[len] != ';') {
      h = (h * 31) + (unsigned char)s[len++];
    }

    // parse decimal number as int
    s = parse_number(&temperature, s + len + 1);

    // probe map until free spot or match
    h = h & (HCAP - 1);
    while (result->map[h] >= 0 && memcmp(result->labels[result->map[h]],
                                         linestart, (size_t)len) != 0) {
      h = (h + 1) & (HCAP - 1);
    }
    c = result->map[h];

    if (c < 0) {
      memcpy(result->labels[result->n], linestart, (size_t)len);
      result->labels[result->n][len] = 0x0;
      result->groups[result->n].label = result->labels[result->n];
      result->groups[result->n].count = 1;
      result->groups[result->n].sum = temperature;
      result->groups[result->n].min = temperature;
      result->groups[result->n].max = temperature;
      result->map[h] = result->n++;
    } else {
      result->groups[c].count += 1;
      result->groups[c].sum += temperature;
      result->groups[c].min = min(result->groups[c].min, temperature);
      result->groups[c].max = max(result->groups[c].max, temperature);
    }
  }

  return (void *)result;
}

void result_to_str(char *dest, const struct Result *result) {
  char buf[128];
  *dest++ = '{';
  for (int i = 0; i < result->n; i++) {
    size_t n = (size_t)sprintf(
        buf, "%s=%.1f/%.1f/%.1f", result->groups[i].label,
        (float)result->groups[i].min / 10.0,
        ((float)result->groups[i].sum / (float)result->groups[i].count) / 10.0,
        (float)result->groups[i].max / 10.0);

    memcpy(dest, buf, n);
    if (i < result->n - 1) {
      memcpy(dest + n, ", ", 2);
      n += 2;
    }

    dest += n;
  }
  *dest++ = '}';
  *dest = 0x0;
}

int main(int argc, char **argv) {
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  char *file = "measurements.txt";
  if (argc > 1) {
    file = argv[1];
  }

  int fd = open(file, O_RDONLY);
  if (!fd) {
    perror("error opening file");
    exit(EXIT_FAILURE);
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("error getting file size");
    exit(EXIT_FAILURE);
  }

  // mmap entire file into memory
  size_t sz = (size_t)sb.st_size;
  const char *data = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    perror("error mmapping file");
    exit(EXIT_FAILURE);
  }

  // distribute work among N worker threads
  pthread_t workers[NTHREADS];
  struct Chunk chunks[NTHREADS];
  size_t chunk_size = sz / (size_t)NTHREADS;
  for (int i = 0; i < NTHREADS; i++) {
    chunks[i].data = data;
    chunks[i].start = chunk_size * (size_t)i;
    chunks[i].end = chunk_size * ((size_t)i + 1);
    pthread_create(&workers[i], NULL, process_chunk, &chunks[i]);
  }

  // wait for all threads to finish
  struct Result *results[NTHREADS];
  for (int i = 0; i < NTHREADS; i++) {
    pthread_join(workers[i], (void *)&results[i]);
  }

  // merge results
  char *label;
  struct Group *b;
  unsigned int h;
  int c;
  struct Result *result = results[0];
  for (int i = 1; i < NTHREADS; i++) {
    for (int j = 0; j < results[i]->n; j++) {
      b = &results[i]->groups[j];
      label = results[i]->labels[j];
      h = hash_probe(result->map, result->labels, label, (int)strlen(label));

      // TODO: Refactor lines below, we can share some logic with process_chunk
      c = result->map[h];
      if (c >= 0) {
        result->groups[c].count += b->count;
        result->groups[c].sum += b->sum;
        result->groups[c].min = min(result->groups[c].min, b->min);
        result->groups[c].max = max(result->groups[c].max, b->max);
      } else {
        // memcpy(&result->groups[result->n], b, sizeof(*b));
        strcpy(result->labels[result->n], label);
        result->groups[result->n].count = b->count;
        result->groups[result->n].sum = b->sum;
        result->groups[result->n].min = b->min;
        result->groups[result->n].max = b->max;
        result->groups[result->n].label = result->labels[result->n];
        result->map[h] = result->n++;
      }
    }
  }

  // sort results alphabetically
  qsort(result->groups, (size_t)result->n, sizeof(struct Group), cmp);

  // prepare output string
  char buf[(1 << 10) * 16];
  result_to_str(buf, result);
  puts(buf);

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

  printf("Runtime inside main = %fms\n", elapsed_time);

  // // clean-up

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  munmap(data, sz);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;                          
  printf("munmap cost = %fms\n", elapsed_time);

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  close(fd);
  for (int i = 0; i < NTHREADS; i++) {
    free(results[i]);
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;                          
  printf("free memory cost = %fms\n", elapsed_time);
  
  // exit(EXIT_SUCCESS);
}

// Runtime inside main = 343.277755ms
// munmap cost = 216.992869ms
// free memory cost = 19.699456ms
// real    0m0.582s
// user    0m29.590s
// sys     0m2.204s

// Dual EPYC 9354 128 threads
// ```
// Runtime inside main = 343.277755ms
// munmap cost = 216.992869ms
// free memory cost = 19.699456ms
// real    0m0.582s
// user    0m29.590s
// sys     0m2.204s
// ```

// AMD 2950X, 32 threads
// ```
// Runtime inside main = 979.854644ms
// munmap cost = 152.057272ms
// free memory cost = 3.055444ms
// real    0m1.137s
// user    0m28.855s
// sys     0m0.734s
// ```

// AMD 2950X, 1 thread
// ```
// Runtime inside main = 18154.117726ms
// munmap cost = 156.046306ms
// free memory cost = 0.126220ms
// real    0m18.312s
// user    0m17.956s
// sys     0m0.348s
// ```
