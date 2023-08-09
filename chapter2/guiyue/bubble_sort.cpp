#include <iostream>
#include <cstdlib>
#include <algorithm>

const int N = 20;

void quick_sort(int* nums, int lhs, int rhs) {
  if (lhs >= rhs) {
    return;
  }
  int mid = (lhs + rhs) / 2;
  int x = nums[mid];
  int l_tmp = lhs - 1;
  int r_tmp = rhs + 1;
  while (l_tmp < r_tmp) {
    do { l_tmp++; }
    while (nums[l_tmp] < x);
    do { r_tmp--; }
    while (nums[r_tmp] > x);
    if (l_tmp < r_tmp) {
      // int tmp_to_swap = nums[l_tmp];
      // nums[l_tmp] = nums[r_tmp];
      // nums[r_tmp] = tmp_to_swap;
      std::swap(nums[l_tmp], nums[r_tmp]);
    }
  }
  quick_sort(nums, lhs, r_tmp);
  quick_sort(nums, r_tmp + 1, rhs);
}

void bubble_sort(int* nums, int lhs, int rhs) {
  for (int i = lhs; i <= rhs; i++) {
    for (int j = lhs; j <= rhs - 1; j++) {
      if (nums[j] > nums[j + 1]) {
        std::swap(nums[j], nums[j + 1]);
      }
    }
  }
}

int main() {
  int nums[N];
  for (int i = 0; i < N; i++) {
    nums[i] = std::rand() % 100;
  }
  for (int i = 0; i < N; i++) {
    std::printf("%d  ", nums[i]);
  }
  std::printf("\n");
  // quick_sort(nums, 0, N - 1);
  bubble_sort(nums, 0, N - 1);
  for (int i = 0; i < N; i++) {
    std::printf("%d  ", nums[i]);
  }
  std::printf("\n");
  return 0;
}