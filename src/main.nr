use dep::std::println;


// quantized arithmetic mul
// ((x_quantized - x_zero_point) / x_scale_reciprocal) * ((y_quantized - y_zero_point) / y_scale_reciprocal) * z_scale_reciprocal + z_zero_point;
fn mul(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let z_quantized =
       1 / (x_scale_reciprocal * y_scale_reciprocal / z_scale_reciprocal) *
       (x_quantized - x_zero_point) *
       (y_quantized - y_zero_point) +
       z_zero_point;
   z_quantized
}

// quantized arithmetic div
// ((x_quantized - x_zero_point) / x_scale_reciprocal) / ((y_quantized - y_zero_point) / y_scale_reciprocal) * z_scale_reciprocal + z_zero_point;
fn div(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let z_quantized =
       1 / ( x_scale_reciprocal / y_scale_reciprocal / z_scale_reciprocal) *
       (x_quantized - x_zero_point) /
       (y_quantized - y_zero_point) +
       z_zero_point;
   z_quantized
}

// quantized arithmetic add
// ((x_quantized - x_zero_point) / x_scale_reciprocal) + ((y_quantized - y_zero_point) / y_scale_reciprocal) * z_scale_reciprocal + z_zero_point;
fn add(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let mut multiplier_x_y: Field = 0;
   let x_scale_100: u8 = (x_scale_reciprocal * 100) as u8;
   let y_scale_100: u8 = (y_scale_reciprocal * 100) as u8;
   if x_scale_100 > y_scale_100 {
       multiplier_x_y = 2 * y_scale_reciprocal;
   } else {
       multiplier_x_y = 2 * x_scale_reciprocal;
   }
   let multiplier_x = 1 / x_scale_reciprocal / multiplier_x_y;
   let multiplier_y = 1 / y_scale_reciprocal / multiplier_x_y;
   let multiplier_x_y_z = multiplier_x_y * z_scale_reciprocal;
   let z_quantized =
       multiplier_x_y_z *
       (multiplier_x * (x_quantized - x_zero_point) + multiplier_y * (y_quantized - y_zero_point)) +
       z_zero_point;
   z_quantized
}

// quantized arithmetic sub
// ((x_quantized - x_zero_point) / x_scale_reciprocal) - ((y_quantized - y_zero_point) / y_scale_reciprocal) * z_scale_reciprocal + z_zero_point;
fn sub(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let mut multiplier_x_y: Field = 0;
   let x_scale_100: u8 = (x_scale_reciprocal * 100) as u8;
   let y_scale_100: u8 = (y_scale_reciprocal * 100) as u8;
   if x_scale_100 > y_scale_100 {
       multiplier_x_y = 2 * y_scale_reciprocal;
   } else {
       multiplier_x_y = 2 * x_scale_reciprocal;
   }
   let multiplier_x = 1 / x_scale_reciprocal / multiplier_x_y;
   let multiplier_y = 1 / y_scale_reciprocal / multiplier_x_y;
   let multiplier_x_y_z = multiplier_x_y * z_scale_reciprocal;
   let z_quantized =
       multiplier_x_y_z *
       (multiplier_x * (x_quantized - x_zero_point) - multiplier_y * (y_quantized - y_zero_point)) +
       z_zero_point;
   z_quantized
}

// x: [-3.0, 0.1, 3.2, -3.0, -0.3, 3.2, -2.0, 0.2, 2.0, -1.0, 0.1, 1.0, -3.2, 1.0, 3.0, -2.9, 0]
// uint8 x_quantized: [7, 131, 255, 7, 115, 255, 47, 135, 207, 87, 131, 167, 0, 167, 247, 11, 127]
#[test]
fn test_mul() {
    // -3.0 * 0.1 = -0.3
    let z_quantized_mul = mul(7, 127, 40, 131, 127, 40, 127, 40);
    println(z_quantized_mul);
    assert (z_quantized_mul == 115);
}
#[test]
fn test_div() {
    // -0.3/0.1 = -3.0
    let z_quantized_div2 = div(115, 127, 40, 131, 127, 40, 127, 40);
    println(z_quantized_div2);
    assert (z_quantized_div2 == 7);
}
#[test]
fn test_add() {
    // -3.0 + 0.1 = -2.9
    let z_quantized_add = add(7, 127, 40, 131, 127, 40, 127, 40);
    println(z_quantized_add);
    assert (z_quantized_add == 11);
}
#[test]
fn test_sub() {
    // 0.1 - 3.0 = -2.9
    let z_quantized_sub = sub(131, 127, 40, 247, 127, 40, 127, 40);
    println(z_quantized_sub);
    assert (z_quantized_sub == 11);
}

// value scope in [-1,1], quantized scope in [0,255]
#[test]
fn test_add_n_n(){
    // negative add negative
    // -0.1 + -0.1 = -0.2
    let z_quantized_add = add(114, 127, 128, 114, 127, 128, 127, 128);
    println(z_quantized_add);
    assert (z_quantized_add == 101);

}

#[test]
fn test_add_n_p(){
    // negative add positive
    // -0.6 + 0.3 = -0.3
    let z_quantized_add = add(50, 127, 128, 165, 127, 128, 127, 128);
    println(z_quantized_add);
    // assert (z_quantized_add == 88);
}

#[test]
fn test_add_p_p(){
    // positive add positive
    // 0.1 + 0.2 = 0.3
    let z_quantized_add = add(140, 127, 128, 153, 127, 128, 127, 128);
    println(z_quantized_add);
    // assert (z_quantized_add == 166);
}
