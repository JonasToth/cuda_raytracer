
inline CUCALL camera::camera(int width, int height, coord origin, coord steering)
    : __origin{origin}
    , __steering{normalize(steering)}
    , __width{width}
    , __height{height}
{}

inline CUCALL camera::camera(int width, int height)
    : camera(width, height, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f})
{}

inline CUCALL ray camera::rayAt(int u, int v) const noexcept {
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

// http://planning.cs.uiuc.edu/node102.html
inline CUCALL void camera::turn(float yaw, float pitch) noexcept {
    using std::sin;
    using std::cos;

    const float roll = 0.f;

    //const float alpha = yaw;
    //const float beta  = pitch;
    //const float gamma = roll;

    const float alpha = roll;
    const float beta  = yaw;
    const float gamma = pitch;

    const float c1 = cos(alpha);
    const float c2 = cos(beta);
    const float c3 = cos(gamma);
    const float s1 = sin(alpha);
    const float s2 = sin(beta);
    const float s3 = sin(gamma);
    
    float dR[9] = {
        c1 * c2,        c1 * s2 * s3 - c3 * s1,         s1 * s3 + c1 * c3 * s2,
        c2 * s1,        c1 * c3 + s1 * s2 * s3,         c3 * s1 * s2 - c1 * s3,
        -s2,            c2 * s3,                        c2 * c3
    };
    const coord dir = __steering;
    __steering = normalize(coord(
        dR[0] * dir.x + dR[1] * dir.y + dR[2] * dir.z,
        dR[3] * dir.x + dR[4] * dir.y + dR[5] * dir.z,
        dR[6] * dir.x + dR[7] * dir.y + dR[8] * dir.z
    ));
}

inline CUCALL void camera::lookAt(coord target) noexcept {
    __steering = normalize(target - __origin);
}


