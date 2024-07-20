struct StyleUniform {
    intensity_factor: f32,
    thickness: f32,
}

struct QuadraticBezier {
    position_0: vec3<f32>,
    position_1: vec3<f32>,
    position_2: vec3<f32>,
}

struct Vertex {
    position_0: vec3<f32>,
    position_1: vec3<f32>,
    position_2: vec3<f32>,
    position: vec3<f32>,
}

struct GeometryVertices {
    vertices: array<vec3<f32>, 10>,
}


@group(0) @binding(2) var<uniform> u_style: StyleUniform;
@group(1) @binding(0) var<storage, read> s_in: array<QuadraticBezier>;
@group(1) @binding(1) var<storage, read_write> s_vertices: array<Vertex>;
@group(1) @binding(2) var<storage, read_write> s_indices: array<u32>;


// PGA utilities
// vector vec4 basis: e1, e2, e3, e0
// bivector mat2x3 basis: e01, e02, e03, e23, e31, e12
// trivector vec4 basis: e032, e013, e021, e123

// Meets
fn plane_wedge_plane(
    vector_0: vec4<f32>,
    vector_1: vec4<f32>,
) -> mat2x3<f32> {
    return mat2x3(vector_0.w * vector_1.xyz - vector_0.xyz * vector_1.w, cross(vector_0.xyz, vector_1.xyz));
}

fn plane_wedge_line(
    vector: vec4<f32>,
    bivector: mat2x3<f32>,
) -> vec4<f32> {
    return vec4(cross(vector.xyz, bivector[0]) - vector.w * bivector[1], dot(vector.xyz, bivector[1]));
}

// Joins
fn line_antiwedge_point(
    bivector: mat2x3<f32>,
    trivector: vec4<f32>,
) -> vec4<f32> {
    let dual_bivector = mat2x3(bivector[1], bivector[0]);
    return -plane_wedge_line(trivector, dual_bivector);
}

fn point_antiwedge_point(
    trivector_0: vec4<f32>,
    trivector_1: vec4<f32>,
) -> mat2x3<f32> {
    let dual_bivector = plane_wedge_plane(trivector_0, trivector_1);
    return mat2x3(dual_bivector[1], dual_bivector[0]);
}

// Expansions
fn plane_dot_line(
    vector: vec4<f32>,
    bivector: mat2x3<f32>,
) -> vec4<f32> {
    return vec4(-cross(vector.xyz, bivector[1]), -dot(vector.xyz, bivector[0]));
}

fn line_dot_point(
    bivector: mat2x3<f32>,
    trivector: vec4<f32>,
) -> vec4<f32> {
    return vec4(-bivector[1] * trivector.w, dot(bivector[1], trivector.xyz));
}

// Misc
fn shift_plane(
    vector: vec4<f32>,
    offset: f32,
) -> vec4<f32> {
    return vec4(vector.xyz, vector.w + offset * length(vector.xyz));
}


fn compute_bounding_geometry_non_collinear(
    quadratic_bezier: QuadraticBezier,
    radius: f32,
) -> GeometryVertices {
    let point_0 = vec4(quadratic_bezier.position_0, 1.0);
    let point_1 = vec4(quadratic_bezier.position_1, 1.0);
    let point_2 = vec4(quadratic_bezier.position_2, 1.0);
    let line_01 = point_antiwedge_point(point_0, point_1);
    let line_12 = point_antiwedge_point(point_1, point_2);
    let line_20 = point_antiwedge_point(point_2, point_0);
    let plane = line_antiwedge_point(line_20, point_1);
    let extended_plane_01 = shift_plane(plane_dot_line(plane, line_01), -radius);
    let extended_plane_12 = shift_plane(plane_dot_line(plane, line_12), -radius);
    let extended_plane_20 = shift_plane(plane_dot_line(plane, line_20), -radius);
    let extended_plane_0 = shift_plane(line_dot_point(line_01, point_0), -radius);
    let extended_plane_2 = shift_plane(line_dot_point(line_12, point_2), radius);
    let upper_plane = shift_plane(plane, radius);
    let lower_plane = shift_plane(plane, -radius);
    let perp_line_1 = plane_wedge_plane(extended_plane_01, extended_plane_12);
    let perp_line_01 = plane_wedge_plane(extended_plane_01, extended_plane_0);
    let perp_line_21 = plane_wedge_plane(extended_plane_12, extended_plane_2);
    let perp_line_02 = plane_wedge_plane(extended_plane_20, extended_plane_0);
    let perp_line_20 = plane_wedge_plane(extended_plane_20, extended_plane_2);
    let upper_point_1 = plane_wedge_line(upper_plane, perp_line_1);
    let upper_point_01 = plane_wedge_line(upper_plane, perp_line_01);
    let upper_point_21 = plane_wedge_line(upper_plane, perp_line_21);
    let upper_point_02 = plane_wedge_line(upper_plane, perp_line_02);
    let upper_point_20 = plane_wedge_line(upper_plane, perp_line_20);
    let lower_point_1 = plane_wedge_line(lower_plane, perp_line_1);
    let lower_point_01 = plane_wedge_line(lower_plane, perp_line_01);
    let lower_point_21 = plane_wedge_line(lower_plane, perp_line_21);
    let lower_point_02 = plane_wedge_line(lower_plane, perp_line_02);
    let lower_point_20 = plane_wedge_line(lower_plane, perp_line_20);
    return GeometryVertices(array(
        upper_point_1.xyz / upper_point_1.w,
        upper_point_01.xyz / upper_point_01.w,
        upper_point_02.xyz / upper_point_02.w,
        upper_point_20.xyz / upper_point_20.w,
        upper_point_21.xyz / upper_point_21.w,
        lower_point_1.xyz / lower_point_1.w,
        lower_point_01.xyz / lower_point_01.w,
        lower_point_02.xyz / lower_point_02.w,
        lower_point_20.xyz / lower_point_20.w,
        lower_point_21.xyz / lower_point_21.w,
    ));
}


fn collinear(
    position_0: vec3<f32>,
    position_1: vec3<f32>,
    position_2: vec3<f32>,
) -> bool {
    return length(cross(position_0 - position_1, position_2 - position_1)) < 1e-6;
}


fn compute_bounding_geometry(
    quadratic_bezier: QuadraticBezier,
    radius: f32,
) -> GeometryVertices {
    if (radius <= 0.0) {
        return GeometryVertices();
    }

    if (!collinear(quadratic_bezier.position_0, quadratic_bezier.position_1, quadratic_bezier.position_2)) {
        return compute_bounding_geometry_non_collinear(quadratic_bezier, radius);
    }

    // `position_0`, `position_1`, `position_2` are collinear.
    // We need a new position to determine a plane.
    // Find two points sitting on edges (it's unlikely, but possible, that `position_1` is not in the segment `position_0` `position_2`).
    let distance_01 = distance(quadratic_bezier.position_0, quadratic_bezier.position_1);
    let distance_12 = distance(quadratic_bezier.position_1, quadratic_bezier.position_2);
    let distance_02 = distance(quadratic_bezier.position_0, quadratic_bezier.position_2);
    let curve_position_0 = select(quadratic_bezier.position_0, quadratic_bezier.position_1, distance_01 < distance_12 && distance_02 < distance_12);
    let curve_position_2 = select(quadratic_bezier.position_2, quadratic_bezier.position_1, distance_12 < distance_01 && distance_02 < distance_01);
    let mid_position = (curve_position_0 + curve_position_2) / 2.0;
    let offset_length = length(curve_position_2 - curve_position_0) / 16.0;

    for (var dim = 0; dim < 3; dim++) {
        var new_position = vec3(0.0);
        new_position[dim] = 1.0;
        if (!collinear(curve_position_0, new_position, curve_position_2)) {
            let curve_position_1 = mid_position + offset_length * normalize(mid_position - new_position);
            return compute_bounding_geometry_non_collinear(QuadraticBezier(curve_position_0, curve_position_1, curve_position_2), radius);
        }
    }
    // In the case we reach here, all three points coincide, so the curve is effectively a single point.
    return GeometryVertices();
}


@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let index = global_id.x;

    let quadratic_bezier = s_in[index];
    var geometry = compute_bounding_geometry(quadratic_bezier, u_style.thickness / 2.0);

    var vertex: Vertex;
    vertex.position_0 = quadratic_bezier.position_0;
    vertex.position_1 = quadratic_bezier.position_1;
    vertex.position_2 = quadratic_bezier.position_2;
    for (var i = 0u; i < 10; i++) {
        vertex.position = geometry.vertices[i];
        s_vertices[10 * index + i] = vertex;
    }

    var geometry_indices: array<u32, 20> = array(0, 0, 1, 4, 2, 3, 8, 4, 9, 0, 5, 1, 6, 2, 7, 8, 6, 9, 5, 5);
    for (var i = 0u; i < 20; i++) {
        s_indices[20 * index + i] = 10 * index + geometry_indices[i];
    }
}
