#include "curand.h"
#include "hitable.h"
#include "sphere.h"

__device__ __constant__ float COOL_RATE = 0.07f;
__device__ __constant__ float GRAVITY = 0.1281f;
class particle : public sphere {
public:
	particle* collider;
	vec3 original_spawn;
	vec3 original_vel;
	vec3 velocity;
	float temperature;
	int mass; 
	bool draw_mode;
	bool enabled;
	__device__ particle(){}
	__device__ particle(vec3 cen, float r, vec3 m, int index,curandState* state)
	{
		center = cen;
		original_spawn = cen;
		radius = r;
		material = m;
		//float xVelocity = 0;
		//float zVelocity = 0;
		float xVelocity = curand_uniform(&state[index]) * 0.1f - 0.05f;
		float zVelocity = curand_uniform(&state[index]) * 0.1f - 0.05f;
		float yVelocity = -0.0533f;
		velocity = vec3(xVelocity, yVelocity, zVelocity);
		original_vel = vec3(xVelocity, yVelocity, zVelocity);
		mass = 1;
		temperature = 1;
		enabled = true;
		draw_mode = true;
		collider = nullptr;
	};

	__device__ bool particle::is_enabled() {
		return enabled;
	}

	__device__ void particle::enable() {
		enabled = true;
	}

	__device__ void particle::switch_mode() {
		draw_mode = !draw_mode;
	}

	__device__ void particle::disable() {
		enabled = false;
	}

	__device__ void particle::reset() {
		center = original_spawn;
		collider = nullptr;
		velocity = original_vel;
		temperature = 1.0f;
		mass = 1;
	}

	__device__ bool particle::collide(particle* p, int p_index, float sum_radii) {
		float delta_x = center.x() - p->center.x();
		float delta_y = center.y() - p->center.y();
		float delta_z = center.z() - p->center.z();
		float dist = ((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
		if (dist < sum_radii * sum_radii) {
			int thisMass = mass;
			int otherMass = p->mass;
			float x_vel = (thisMass * velocity.x()) + (otherMass * p->velocity.x());
			float z_vel = (thisMass * velocity.z()) + (otherMass * p->velocity.z());
			int denominator = thisMass + otherMass;
			velocity = vec3(x_vel / denominator, -GRAVITY, z_vel / denominator);
			mass += p->mass;
			p->mass = 0;
			p->center = vec3(p_index, 100, 1000);
			p->velocity = vec3(0, 0, 0);
			p->collider = this;
			return true;
		}
		return false;
	}

	__device__ void particle::invert_x() {

		float newX = -velocity.x();
		velocity = vec3(newX, velocity.y(), velocity.z());
	}

	__device__ void particle::invert_z() {

		float newZ = -velocity.z();
		velocity = vec3(velocity.x(), velocity.y(), newZ);
	}

	__device__ void particle::invert_y() {

		float newY = -velocity.y();
		velocity = vec3(velocity.x(), newY, velocity.z());
	}

	__device__ void particle::temperature_tick() {
		temperature -= 0.1666f * COOL_RATE * (1/mass);
		temperature = max(temperature, 0.0f);
		if (draw_mode) {
			float red_component = temperature;
			float blue_component = 1.0f-temperature;
			material = vec3(blue_component,0.0f,red_component);
		}
		else {
			float factor = mass / 5.0f;
			float red_component = factor;
			float blue_component = 1.0f-factor;
			material = vec3(blue_component, 0.0f, red_component);
		}
	}

	__device__ void particle::movement_tick() {
		float newX = center.x() + 0.1666f * velocity.x();
		velocity = newX > 0.5f || newX < -0.5f ? velocity * vec3(-1, 1, 1) : velocity;
		float newY = center.y() + 0.16666f * -GRAVITY * mass;
		//float newY = center.y() + 0.16666f * velocity.y();
		float newZ = center.z() + 0.16666f * velocity.z();
		velocity = newZ > 0.5f || newZ < -0.5f ? velocity * vec3(1, 1, -1) : velocity;
		center = vec3(newX, newY, newZ);
	}
};
