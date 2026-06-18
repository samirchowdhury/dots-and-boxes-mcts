(module
  (memory (export "memory") 1)

  (func (export "select_child")
    (param $visits_ptr i32)
    (param $value_sums_ptr i32)
    (param $priors_ptr i32)
    (param $count i32)
    (param $parent_visits i32)
    (param $c_puct f64)
    (result i32)
    (local $index i32)
    (local $best_index i32)
    (local $child_visits i32)
    (local $best_score f64)
    (local $score f64)
    (local $q f64)
    (local $u f64)
    (local $sqrt_parent f64)

    (local.set $index (i32.const 0))
    (local.set $best_index (i32.const 0))
    (local.set $best_score (f64.const -1.7976931348623157e308))
    (local.set $sqrt_parent
      (f64.sqrt
        (f64.convert_i32_s
          (select
            (local.get $parent_visits)
            (i32.const 1)
            (i32.gt_s (local.get $parent_visits) (i32.const 1))))))

    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $index) (local.get $count)))

        (local.set $child_visits
          (i32.load
            (i32.add
              (local.get $visits_ptr)
              (i32.mul (local.get $index) (i32.const 4)))))

        (if
          (i32.eqz (local.get $child_visits))
          (then
            (local.set $q (f64.const 0)))
          (else
            (local.set $q
              (f64.div
                (f64.load
                  (i32.add
                    (local.get $value_sums_ptr)
                    (i32.mul (local.get $index) (i32.const 8))))
                (f64.convert_i32_s (local.get $child_visits))))))

        (local.set $u
          (f64.div
            (f64.mul
              (f64.mul
                (local.get $c_puct)
                (f64.load
                  (i32.add
                    (local.get $priors_ptr)
                    (i32.mul (local.get $index) (i32.const 8)))))
              (local.get $sqrt_parent))
            (f64.convert_i32_s (i32.add (local.get $child_visits) (i32.const 1)))))

        (local.set $score (f64.add (local.get $q) (local.get $u)))

        (if
          (f64.gt (local.get $score) (local.get $best_score))
          (then
            (local.set $best_score (local.get $score))
            (local.set $best_index (local.get $index))))

        (local.set $index (i32.add (local.get $index) (i32.const 1)))
        (br $loop)))

    (local.get $best_index))
)
