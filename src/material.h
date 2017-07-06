#ifndef MATERIAL_H_YZ53U2I4
#define MATERIAL_H_YZ53U2I4


/// Implement the phong reflection model
struct material {
    // See https://en.wikipedia.org/wiki/Phong_reflection_model
    // for each coefficient
    float ks;     ///< specular reflection
    float kd;     ///< diffuse reflection
    float ka;     ///< ambient reflection
    float alpha;  ///< shininess constant

};


#endif /* end of include guard: MATERIAL_H_YZ53U2I4 */
