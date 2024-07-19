struct CameraUniform {
    projection_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
};

struct ModelUniform {
    model_matrix: mat4x4<f32>,
};

struct StyleUniform {
    color: vec3<f32>,
    opacity: f32,
    thickness: f32,
};

struct Vertex {
    @location(0) position_0: vec3<f32>,
    @location(1) position_1: vec3<f32>,
    @location(2) position_2: vec3<f32>,
    @location(3) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_position_0: vec3<f32>,
    @location(1) view_position_1: vec3<f32>,
    @location(2) view_position_2: vec3<f32>,
    @location(3) view_position: vec3<f32>,
};


@group(0) @binding(0) var<uniform> u_camera: CameraUniform;
@group(0) @binding(1) var<uniform> u_model: ModelUniform;
@group(0) @binding(2) var<uniform> u_style: StyleUniform;


fn get_view_position(
    view_model_matrix: mat4x4<f32>,
    position: vec3<f32>,
) -> vec3<f32> {
    let view_position_homogeneous = view_model_matrix * vec4(position, 1.0);
    return view_position_homogeneous.xyz / view_position_homogeneous.w;
}


@vertex
fn vs_main(
    in: Vertex,
) -> VertexOutput {
    let view_model_matrix = u_camera.view_matrix * u_model.model_matrix;
    let view_position_0 = get_view_position(view_model_matrix, in.position_0);
    let view_position_1 = get_view_position(view_model_matrix, in.position_1);
    let view_position_2 = get_view_position(view_model_matrix, in.position_2);
    let view_position = get_view_position(view_model_matrix, in.position);
    return VertexOutput(
        u_camera.projection_matrix * vec4(view_position, 1.0),
        view_position_0,
        view_position_1,
        view_position_2,
        view_position,
    );
}


struct SmallVec {
    count: u32,
    values: array<f32, 4>,
}

// Determines how the sign of a polynomial behaves.
// Roots should be sorted in *descending* order.
struct PolynomialHomotopy {
    leading_coefficient_positive: bool,
    roots: SmallVec,
}


fn cbrt(
    x: f32,
) -> f32 {
    return select(sign(x) * pow(abs(x), 1.0 / 3.0), 0.0, x == 0.0);
}

fn append(
    vec_ptr: ptr<function, SmallVec>,
    new_value: f32,
) {
    (*vec_ptr).values[(*vec_ptr).count] = new_value;
    (*vec_ptr).count++;
}

fn insertion_sort_descending(
    vec_ptr: ptr<function, SmallVec>,
) {
    let count = (*vec_ptr).count;
    let values_ptr = &(*vec_ptr).values;
    for (var i = 1u; i < count; i++) {
        var j = i;
        let value = (*values_ptr)[j];
        while (j > 0 && (*values_ptr)[j - 1] < value) {
            (*values_ptr)[j] = (*values_ptr)[j - 1];
            j--;
        }
        (*values_ptr)[j] = value;
    }
}

fn solve_monic_quadratic(
    c: f32,
    b: f32,
) -> SmallVec {
    var roots = SmallVec();
    // Solving x^2 + b x + c = 0 (#roots: 0 / 2)
    // => t^2 + p = 0, t = x - u
    let u = -b / 2.0;
    let p = c - u * u;

    if (p < 0.0) {
        let sqrt_neg_p = sqrt(-p);
        append(&roots, u + sqrt_neg_p);
        append(&roots, u - sqrt_neg_p);
    } else if (p == 0.0) {
        append(&roots, u);
        append(&roots, u);
    }
    return roots;
}

// Reference: https://en.wikipedia.org/wiki/Cubic_equation
fn solve_monic_cubic(
    d: f32,
    c: f32,
    b: f32,
) -> SmallVec {
    var roots = SmallVec();
    // Solving x^3 + b x^2 + c x + d = 0 (#roots: 1 / 3)
    // => t^3 + p t + q = 0, t = x - u
    let u = -b / 3.0;
    let p = c - 3.0 * u * u;
    let q = d + u * c - 2.0 * u * u * u;

    let p_over_3 = p / 3.0;
    let q_over_2 = q / 2.0;
    let discriminant = p_over_3 * p_over_3 * p_over_3 + q_over_2 * q_over_2;
    if (discriminant < 0.0) {
        let sqrt_neg_p_over_3 = sqrt(-p_over_3);
        let theta = acos(clamp(q_over_2 / (p_over_3 * sqrt_neg_p_over_3), -1.0, 1.0)) / 3.0;
        let tau_over_3 = 2.0 * acos(-1.0) / 3.0;
        for (var k = 0; k < 3; k++) {
            append(&roots, u + 2.0 * sqrt_neg_p_over_3 * cos(theta - f32(k) * tau_over_3));
        }
    } else if (discriminant == 0.0) {
        let cbrt_neg_q_over_2 = cbrt(-q_over_2);
        append(&roots, u + 2.0 * cbrt_neg_q_over_2);
        append(&roots, u - cbrt_neg_q_over_2);
        append(&roots, u - cbrt_neg_q_over_2);
    } else {
        let sqrt_discriminant = sqrt(discriminant);
        append(&roots, u + cbrt(-q_over_2 + sqrt_discriminant) + cbrt(-q_over_2 - sqrt_discriminant));
    }

    // Newton iteration.
    for (var i = 0u; i < roots.count; i++) {
        var root = roots.values[i];
        for (var iteration_step = 0u; iteration_step < 2; iteration_step++) {
            root -= (d + (c + (b + root) * root) * root) / (c + (2.0 * b + 3.0 * root) * root);
        }
        roots.values[i] = root;
    }
    return roots;
}

// Reference: https://en.wikipedia.org/wiki/Quartic_equation
fn solve_monic_quartic(
    e: f32,
    d: f32,
    c: f32,
    b: f32,
) -> SmallVec {
    var roots = SmallVec();
    // Solving x^4 + b x^3 + c x^2 + d x + e = 0 (#roots: 0 / 2 / 4)
    // => t^4 + p t^2 + q t + r = 0, t = x - u
    let u = -b / 4.0;
    let p = c - 6.0 * u * u;
    let q = d + 2.0 * u * c - 8.0 * u * u * u;
    let r = e + u * d + u * u * c - 3.0 * u * u * u * u;

    let y = solve_monic_cubic(
        -q * q,
        p * p - 4.0 * r,
        2.0 * p,
    ).values[0];
    if (y > 0.0) {
        let sqrt_y = sqrt(y);
        var quadratic_roots_0 = solve_monic_quadratic(
            (y + p - q / sqrt_y) / 2.0,
            sqrt_y,
        );
        var quadratic_roots_1 = solve_monic_quadratic(
            (y + p + q / sqrt_y) / 2.0,
            -sqrt_y,
        );
        for (var i = 0u; i < quadratic_roots_0.count; i++) {
            append(&roots, u + quadratic_roots_0.values[i]);
        }
        for (var i = 0u; i < quadratic_roots_1.count; i++) {
            append(&roots, u + quadratic_roots_1.values[i]);
        }
    } else if (y == 0.0) {
        var quadratic_roots = solve_monic_quadratic(
            r,
            p,
        );
        for (var i = 0u; i < quadratic_roots.count; i++) {
            let quadratic_root = quadratic_roots.values[i];
            if (quadratic_root > 0.0) {
                let sqrt_quadratic_root = sqrt(quadratic_root);
                append(&roots, u + sqrt_quadratic_root);
                append(&roots, u - sqrt_quadratic_root);
            } else if (quadratic_root == 0.0) {
                append(&roots, u);
                append(&roots, u);
            }
        }
    }

    // Newton iteration.
    for (var i = 0u; i < roots.count; i++) {
        var root = roots.values[i];
        for (var iteration_step = 0u; iteration_step < 2; iteration_step++) {
            root -= (e + (d + (c + (b + root) * root) * root) * root) / (d + (2.0 * c + (3.0 * b + 4.0 * root) * root) * root);
        }
        roots.values[i] = root;
    }
    return roots;
}


fn positive_intersection(
    homotopy_ptr_0: ptr<function, PolynomialHomotopy>,
    homotopy_ptr_1: ptr<function, PolynomialHomotopy>,
) -> PolynomialHomotopy {
    let roots_ptr_0 = &(*homotopy_ptr_0).roots;
    let roots_ptr_1 = &(*homotopy_ptr_1).roots;
    let count_0 = (*roots_ptr_0).count;
    let count_1 = (*roots_ptr_1).count;
    var root_index_0 = 0u;
    var root_index_1 = 0u;
    var positive_0 = (*homotopy_ptr_0).leading_coefficient_positive;
    var positive_1 = (*homotopy_ptr_1).leading_coefficient_positive;
    let leading_coefficient_positive = positive_0 && positive_1;
    var roots = SmallVec();
    loop {
        var max_root_flag = 2u;
        var max_root = 0.0;
        if (root_index_0 < count_0) {
            let root = (*roots_ptr_0).values[root_index_0];
            if (max_root_flag == 2 || root > max_root) {
                max_root_flag = 0u;
                max_root = root;
            }
        }
        if (root_index_1 < count_1) {
            let root = (*roots_ptr_1).values[root_index_1];
            if (max_root_flag == 2 || root > max_root) {
                max_root_flag = 1u;
                max_root = root;
            }
        }
        if (max_root_flag == 0) {
            root_index_0++;
            positive_0 = !positive_0;
            if (positive_1) {
                append(&roots, max_root);
            }
        } else if (max_root_flag == 1) {
            root_index_1++;
            positive_1 = !positive_1;
            if (positive_0) {
                append(&roots, max_root);
            }
        } else {
            break;
        }
    }
    return PolynomialHomotopy(leading_coefficient_positive, roots);
}


fn get_intensity(
    view_position_0: vec3<f32>,
    view_position_1: vec3<f32>,
    view_position_2: vec3<f32>,
    view_position: vec3<f32>,
    radius: f32,
) -> f32 {
    let ray_direction = normalize(view_position);
    let view_position_0_scaled = view_position_0 / radius;
    let view_position_1_scaled = view_position_1 / radius;
    let view_position_2_scaled = view_position_2 / radius;
    let quadratic_bezier = mat3x3(
        dot(view_position_0_scaled, ray_direction) * ray_direction - view_position_0_scaled,
        dot(view_position_1_scaled, ray_direction) * ray_direction - view_position_1_scaled,
        dot(view_position_2_scaled, ray_direction) * ray_direction - view_position_2_scaled,
    ) * mat3x3(
         1.0,  0.0,  0.0,
        -2.0,  2.0,  0.0,
         1.0, -2.0,  1.0,
    );
    let linear_bezier = mat2x3(
        view_position_1_scaled - view_position_0_scaled,
        view_position_2_scaled - view_position_1_scaled,
    ) * mat2x2(
         1.0,  0.0,
        -1.0,  1.0,
    );

    // Calculate the integral of 3/2 quartic(t) sqrt(q0 + q1 t + q2 t^2)
    // in interval (0, 1), intersected with intervals where the quartic expression evaluates to positive.
    var coefficients = array(
        -dot(quadratic_bezier[0], quadratic_bezier[0]) + 1.0,
        -2.0 * dot(quadratic_bezier[0], quadratic_bezier[1]),
        -(dot(quadratic_bezier[1], quadratic_bezier[1]) + 2.0 * dot(quadratic_bezier[0], quadratic_bezier[2])),
        -2.0 * dot(quadratic_bezier[1], quadratic_bezier[2]),
        -dot(quadratic_bezier[2], quadratic_bezier[2]),
    );
    let q0 = dot(linear_bezier[0], linear_bezier[0]);
    let q1 = 2.0 * dot(linear_bezier[0], linear_bezier[1]);
    let q2 = dot(linear_bezier[1], linear_bezier[1]);

    var leading_coefficient = 0.0;
    var roots = SmallVec();
    // This quartic expression will be one of:
    if (abs(coefficients[4]) >= 1e-4) {
        // - a quartic polynomial with a negative leading coefficient (#roots: 0 / 2 / 4),
        leading_coefficient = coefficients[4];
        roots = solve_monic_quartic(
            coefficients[0] / leading_coefficient,
            coefficients[1] / leading_coefficient,
            coefficients[2] / leading_coefficient,
            coefficients[3] / leading_coefficient,
        );
    } else if (abs(coefficients[2]) >= 1e-4) {
        // - a quadratic polynomial with a negative leading coefficient (#roots: 0 / 2),
        leading_coefficient = coefficients[2];
        roots = solve_monic_quadratic(
            coefficients[0] / leading_coefficient,
            coefficients[1] / leading_coefficient,
        );
    } else {
        // - a constant polynomial (#roots: 0).
        leading_coefficient = coefficients[0];
    }
    insertion_sort_descending(&roots);
    var homotopy = PolynomialHomotopy(leading_coefficient > 0.0, roots);
    var domain_roots = SmallVec();
    append(&domain_roots, 1.0);
    append(&domain_roots, 0.0);
    var domain_homotopy = PolynomialHomotopy(false, domain_roots);
    homotopy = positive_intersection(&homotopy, &domain_homotopy);
    if (homotopy.roots.count == 0) {
        discard;
    }

    var integral_sum = 0.0;
    if (q2 >= 1e-4) {
        // q0 + q1 t + q2 t^2 = q2 ((t - sigma)^2 - delta)
        let sigma = -q1 / (2.0 * q2);
        let delta = -q0 / q2 + sigma * sigma;
        let sqrt_q2 = sqrt(q2);

        for (var i = 0u; i <= 4; i++) {
            var coefficient = 0.0;
            var coefficient_factor = 1.0;
            for (var j = i; j <= 4; j++) {
                coefficient += coefficient_factor * coefficients[j];
                coefficient_factor *= f32(j + 1) / f32(j - i + 1) * sigma;
            }
            coefficients[i] = coefficient;
        }
        for (var i = 0u; i <= 4; i++) {
            var coefficient = 0.0;
            var coefficient_factor = 1.0 / f32(i + 2);
            for (var j = i; j <= 4; j += 2u) {
                coefficient += coefficient_factor * coefficients[j];
                coefficient_factor *= f32(j + 1) / f32(j + 4) * delta;
            }
            coefficients[i] = coefficient;
        }

        var sign = 1.0;
        for (var i = 0u; i < homotopy.roots.count; i++) {
            let t = homotopy.roots.values[i] - sigma;
            var polynomial_value = 0.0;
            for (var degree = 4; degree > 0; degree--) {
                polynomial_value *= t;
                polynomial_value += coefficients[degree];
            }
            let sqrt_t_squared_minus_delta = select(sqrt(t * t - delta), abs(t), abs(delta) < 1e-4);
            let integral_value = sqrt_q2 * (
                polynomial_value * sqrt_t_squared_minus_delta * sqrt_t_squared_minus_delta * sqrt_t_squared_minus_delta
                + coefficients[0] * (t * sqrt_t_squared_minus_delta - select(delta * asinh(t * inverseSqrt(-delta)), 0.0, abs(delta) < 1e-4))
            );
            integral_sum += sign * integral_value;
            sign *= -1.0;
        }
    } else if (q0 >= 1e-4) {
        let sqrt_q0 = sqrt(q0);

        for (var i = 0u; i <= 4; i++) {
            coefficients[i] /= f32(i + 1);
        }

        var sign = 1.0;
        for (var i = 0u; i < homotopy.roots.count; i++) {
            let t = homotopy.roots.values[i];
            var polynomial_value = 0.0;
            for (var degree = 4; degree > 0; degree--) {
                polynomial_value *= t;
                polynomial_value += coefficients[degree];
            }
            let integral_value = sqrt_q0 * (polynomial_value * t * t + coefficients[0] * t);
            integral_sum += sign * integral_value;
            sign *= -1.0;
        }
    }

    return (3.0 / 2.0) * integral_sum;
}


@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) f32 {
    return get_intensity(
        in.view_position_0,
        in.view_position_1,
        in.view_position_2,
        in.view_position,
        u_style.thickness / 2.0,
    );
}
