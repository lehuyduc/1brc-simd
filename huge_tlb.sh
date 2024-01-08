#!/bin/bash

# Script to mount HugeTLBfs with 2GB page size
# Requires sudo privileges

mount_point="./hugetlb"
page_size="2G"

# Create a mount point
sudo mkdir -p $mount_point

# Mount HugeTLBfs with 2GB page size
sudo mount -t hugetlbfs -o pagesize=$page_size nodev $mount_point

echo "HugeTLBfs mounted at $mount_point with 2GB page size"
