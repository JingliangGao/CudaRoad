# Reduce Case

## Information
| case                           |  information                                 |
| ------------------------------ | -------------------------------------------- |
| doc                            | document                                     |
| third_party                    | reference code                               |
| 0_prepare_env                  | prepare CUDA environment                     |
| 1_reduce_global_memory         | 'reduce' op using global memory              |
| 2_reduce_shared_memory         | 'reduce' op using shared memory              |
| 3_reduce_no_divergence_branch  | 'reduce' op decreasing wrap divergence       |
| 4_reduce_no_bank_conflict      | 'reduce' op without bank conflict            |
| 5_reduce_add_during_load_PlanA | 'reduce' op add during load process (Plan A) |
| 6_reduce_add_during_load_PlanB | 'reduce' op add during load process (Plan B) |
| 7_reduce_unroll_last_dim       | 'reduce' op unroll at last dim               |
| 8_reduce_complete_unroll       | 'reduce' op unroll for cycle                 |


# Build
```bash
chmod +x build-for-debug.sh && ./build-for-debug.sh 
```

# Reference
- [【CUDA】Reduce规约求和（已完结~）](https://www.bilibili.com/video/BV1HvBSY2EJW?spm_id_from=333.788.videopod.episodes&vd_source=e48ec78ddbe68f6cb5ea86829024a133&p=5)
- [How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)
- [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)