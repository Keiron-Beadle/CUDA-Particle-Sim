#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        hitable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        vec3 temp_material(0, 0, 0);
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            hitable* m = list[1];
            bool result = list[i]->hit(r, t_min, closest_so_far, temp_rec);
            if (result)
            {
                temp_material = list[i]->material;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                rec.color = temp_material;
            }
        }
        return true;
}

#endif
