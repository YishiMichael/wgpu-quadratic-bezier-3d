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

struct Vector3DPGA {
    bulk: vec3<f32>,    // basis: e1, e2, e3
    weight: f32,        // basis: e0
}

struct Bivector3DPGA {
    bulk: vec3<f32>,    // basis: e23, e31, e12
    weight: vec3<f32>,  // basis: e01, e02, e03
}

struct Trivector3DPGA {
    bulk: f32,          // basis: e123
    weight: vec3<f32>,  // basis: e032, e013, e021
}

// Duals
fn plane_dual(
    vector: Vector3DPGA
) -> Trivector3DPGA {
    return Trivector3DPGA(vector.weight, vector.bulk);
}

fn line_dual(
    bivector: Bivector3DPGA
) -> Bivector3DPGA {
    return Bivector3DPGA(bivector.weight, bivector.bulk);
}

fn point_dual(
    trivector: Trivector3DPGA
) -> Vector3DPGA {
    return Vector3DPGA(trivector.weight, trivector.bulk);
}

// Meets
fn plane_wedge_plane(
    vector_0: Vector3DPGA,
    vector_1: Vector3DPGA,
) -> Bivector3DPGA {
    return Bivector3DPGA(cross(vector_0.bulk, vector_1.bulk), vector_0.weight * vector_1.bulk - vector_0.bulk * vector_1.weight);
}

fn plane_wedge_line(
    vector: Vector3DPGA,
    bivector: Bivector3DPGA,
) -> Trivector3DPGA {
    return Trivector3DPGA(dot(vector.bulk, bivector.bulk), cross(vector.bulk, bivector.weight) - vector.weight * bivector.bulk);
}

// Joins
fn line_antiwedge_point(
    bivector: Bivector3DPGA,
    trivector: Trivector3DPGA,
) -> Vector3DPGA {
    return point_dual(plane_wedge_line(point_dual(trivector), line_dual(bivector)));
}

fn point_antiwedge_point(
    trivector_0: Trivector3DPGA,
    trivector_1: Trivector3DPGA,
) -> Bivector3DPGA {
    return line_dual(plane_wedge_plane(point_dual(trivector_0), point_dual(trivector_1)));
}

// Expansions
fn plane_dot_line(
    vector: Vector3DPGA,
    bivector: Bivector3DPGA,
) -> Vector3DPGA {
    return Vector3DPGA(-cross(vector.bulk, bivector.bulk), -dot(vector.bulk, bivector.weight));
}

fn line_dot_point(
    bivector: Bivector3DPGA,
    trivector: Trivector3DPGA,
) -> Vector3DPGA {
    return Vector3DPGA(-bivector.bulk * trivector.bulk, dot(bivector.bulk, trivector.weight));
}

// Misc
fn shift_plane(
    vector: Vector3DPGA,
    offset: f32,
) -> Vector3DPGA {
    return Vector3DPGA(vector.bulk, vector.weight + offset * length(vector.bulk));
}


fn compute_bounding_geometry_non_collinear(
    quadratic_bezier: QuadraticBezier,
    radius: f32,
) -> GeometryVertices {
    let point_0 = Trivector3DPGA(1.0, quadratic_bezier.position_0);
    let point_1 = Trivector3DPGA(1.0, quadratic_bezier.position_1);
    let point_2 = Trivector3DPGA(1.0, quadratic_bezier.position_2);
    let line_01 = point_antiwedge_point(point_0, point_1);
    let line_12 = point_antiwedge_point(point_1, point_2);
    let line_20 = point_antiwedge_point(point_2, point_0);
    let plane = line_antiwedge_point(line_20, point_1);
    let extended_plane_01 = shift_plane(plane_dot_line(plane, line_01), radius);
    let extended_plane_12 = shift_plane(plane_dot_line(plane, line_12), radius);
    let extended_plane_20 = shift_plane(plane_dot_line(plane, line_20), radius);
    let extended_plane_0 = shift_plane(line_dot_point(line_01, point_0), -radius);
    let extended_plane_2 = shift_plane(line_dot_point(line_12, point_2), radius);
    let upper_plane = shift_plane(plane, -radius);
    let lower_plane = shift_plane(plane, radius);
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
        upper_point_1.weight / upper_point_1.bulk,
        upper_point_01.weight / upper_point_01.bulk,
        upper_point_02.weight / upper_point_02.bulk,
        upper_point_20.weight / upper_point_20.bulk,
        upper_point_21.weight / upper_point_21.bulk,
        lower_point_1.weight / lower_point_1.bulk,
        lower_point_01.weight / lower_point_01.bulk,
        lower_point_02.weight / lower_point_02.bulk,
        lower_point_20.weight / lower_point_20.bulk,
        lower_point_21.weight / lower_point_21.bulk,
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
