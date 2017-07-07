
CUCALL camera::camera(int width, int height, coord origin, coord steering)
    : __origin{origin}
    , __steering{normalize(steering)}
    , __width{width}
    , __height{height}
{}

CUCALL camera::camera(int width, int height)
    : camera(width, height, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f})
{}

CUCALL ray camera::rayAt(int u, int v) const noexcept {
    // imeplement pinhole model
    const float focal_length = 360.f;
    const float fx = focal_length, fy = focal_length;
    const int cx = __width / 2;
    const int cy = __height / 2;

    const float x = (u - cx) / fx - (v - cy) / fy;
    const float y = (v - cy) / fy;

    // calculate the direction for camera coordinates 
    const float coeff = std::sqrt(1 + x*x + y*y) / (x*x + y*y + 1);
    const coord dir(coeff * x, coeff * y, coeff);

    float R[9];
    rotation({0.f, 0.f, 0.1f}, __steering, R);

    // transform that direction into world coordinates (steering vector should do that)
    const coord rotated_dir(
            R[0] * dir.x + R[1] * dir.y + R[2] * dir.z,
            R[3] * dir.x + R[4] * dir.y + R[5] * dir.z,
            R[6] * dir.x + R[7] * dir.y + R[8] * dir.z
    );

    return ray(__origin, rotated_dir);
}

// beta => mouse x movement
// gamma=> mouse y movement
// http://planning.cs.uiuc.edu/node102.html
CUCALL void camera::swipe(float alpha, float beta, float gamma) noexcept {
    using std::sin;
    using std::cos;
    float dR[9] = {
         cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma), sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
         sin(alpha) * sin(beta), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma), sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
        -sin(beta),              cos(beta) * sin(gamma),                                        cos(beta) * cos(gamma)
    };
    const coord dir = __steering;
    __steering = normalize(coord(
        dR[0] * dir.x + dR[1] * dir.y + dR[2] * dir.z,
        dR[3] * dir.x + dR[4] * dir.y + dR[5] * dir.z,
        dR[6] * dir.x + dR[7] * dir.y + dR[8] * dir.z
    ));
}

CUCALL void camera::lookAt(coord target) noexcept {
    __steering = normalize(target - __origin);
}


