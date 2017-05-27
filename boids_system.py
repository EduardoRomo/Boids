#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.spatial import cKDTree as kd_tree

colors = np.array([(255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 165, 0),
    (255, 20, 157),
    (138, 43, 226),
    (0, 191, 255),
    (127, 255, 212)])


class boids_system(object):
    def __init__(self, num_boids, max_num_boids, radius, neighborhood_size, screen_size):
        self.num_boids = num_boids
        self.max_num_boids = max_num_boids
        self.radius = radius
        self.neighborhood_size = neighborhood_size
        self.screen_size = screen_size

        self.positions = np.random.uniform(radius, screen_size-radius, (max_num_boids, 2))

        theta = np.random.uniform(0, 2*math.pi, max_num_boids)
        r = np.random.uniform(1.0, 2.0, max_num_boids)
        x, y = r*np.cos(theta), r*np.sin(theta)

        self.angles = theta
        self.velocities = np.array(list(zip(x, y)))
        self.max_speed = 2.0

        self.colors = colors[np.random.randint(0, len(colors), max_num_boids)]
        self.display_colors = self.colors

        self.cohesion_weight = 1.0
        self.alignment_weight = 1.0
        self.separation_weight = 1.0
        self.obstacles_repulsion_weight = 5.0
        self.attractors_cohesion_weight = 0.4
        self.attractors_repulsion_weight = 0.3

        self.num_obstacles = 0
        self.max_num_obstacles = 50
        self.obstacles = np.empty((self.max_num_obstacles, 2))
        self.obstacles = self.obstacles.astype(np.int32)

        self.num_attractors = 0
        self.max_num_attractors = 50
        self.attractors = np.empty((self.max_num_attractors, 2))
        self.attractors = self.attractors.astype(np.int32)


    def get_cohesion(self):
        num_boids = self.num_boids
        cohesion = np.zeros((num_boids, 2))

        for i in range(num_boids):
            num_neighbors = len(self.close_pairs[i])

            if num_neighbors > 0:
                mass_center = np.mean(self.positions[self.close_pairs[i]], axis=0)
                cohesion[i] = mass_center - self.positions[i]
                norm = np.linalg.norm(cohesion[i])
                cohesion[i] /= norm

        return cohesion


    def get_alignment(self):
        num_boids = self.num_boids
        alignment = np.zeros((num_boids, 2))

        for i in range(num_boids):
            num_neighbors = len(self.close_pairs[i])

            if num_neighbors > 0: 
                alignment[i] = np.sum(self.velocities[self.close_pairs[i]], axis=0)
                norm = np.linalg.norm(alignment[i])
                alignment[i] /= norm

        return alignment


    def get_separation(self):
        num_boids = self.num_boids
        separation = np.zeros((num_boids, 2))

        for i in range(num_boids):
            num_neighbors = len(self.close_pairs[i])

            if num_neighbors > 0:
                separation[i] += num_neighbors*self.positions[i] - \
                        np.sum(self.positions[self.close_pairs[i]], axis=0)
                norm = np.linalg.norm(separation[i])
                separation[i] /= norm

        return separation


    def get_random_velocity(self):
        velocities = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            if np.random.rand() < 0.05:
                theta = np.random.uniform(2, 2*math.pi)
                r = np.random.uniform(0, 0.5)
                velocities[i] = r*np.array([math.cos(theta), math.sin(theta)])

        return velocities


    def update_colors(self):
        num_boids = self.num_boids
        display_colors = np.copy(self.colors[:num_boids])

        for k in range(5):
            for i in range(num_boids):
                num_neighbors = len(self.close_pairs[i])

                if num_neighbors > 0:
                    display_colors[i] = np.mean(display_colors[self.close_pairs[i]], axis=0)

        self.display_colors = display_colors


    def limit_positions(self):
        lim_min, lim_max = 0, self.screen_size
        v = 5

        for i, position in enumerate(self.positions[:self.num_boids]):
            if position[0] < lim_min:
                self.velocities[i, 0] = v
            elif position[0] > lim_max:
                self.velocities[i, 0] = -v

            if position[1] < lim_min:
                self.velocities[i, 1] = v
            elif position[1] > lim_max:
                self.velocities[i, 1] = -v


    def limit_velocity(self):
        for i, velocity in enumerate(self.velocities[:self.num_boids]):
            norm = np.linalg.norm(velocity)
            if norm > self.max_speed:
                velocity *= self.max_speed/norm
                self.velocities[i] = velocity


    def get_obstacles_repulsion(self):
        num_boids = self.num_boids
        separation = np.zeros((num_boids, 2))

        for i in range(num_boids):
            num_close_obstacles = len(self.close_pairs_obstacles[i])

            if num_close_obstacles > 0:
                mass_center = np.mean(self.obstacles[self.close_pairs_obstacles[i]], axis=0)
                separation[i] = self.positions[i] - mass_center
                norm = np.linalg.norm(separation[i])
                separation[i] /= norm

        return separation


    def get_attractors_cohesion(self):
        num_boids = self.num_boids
        cohesion = np.zeros((num_boids, 2))

        for i in range(num_boids):
            num_close_attractors = len(self.close_pairs_attractors[i])

            if num_close_attractors > 0:
                mass_center = np.mean(self.attractors[self.close_pairs_attractors[i]], axis=0)
                cohesion[i] = mass_center - self.positions[i]
                norm = np.linalg.norm(cohesion[i])
                cohesion[i] /= norm

        return cohesion

    def get_attractors_repulsion(self):
        num_boids = self.num_boids
        repulsion = np.zeros((num_boids, 2))

        for i in range(num_boids):
            num_close_attractors = len(self.closest_pairs_attractors[i])

            if num_close_attractors > 0:
                mass_center = np.mean(self.attractors[self.closest_pairs_attractors[i]], axis=0)
                repulsion[i] = self.positions[i] - mass_center
                norm = np.linalg.norm(repulsion[i])
                repulsion[i] /= norm

        return repulsion


    def update(self):
        num_boids = self.num_boids
        num_obstacles = self.num_obstacles
        num_attractors = self.num_attractors

        positions = self.positions[:num_boids]

        positions += self.velocities[:num_boids]
        # self.limit_positions()

        self.positions[:num_boids] = np.fmod(positions + self.screen_size,
                self.screen_size)

        tree = kd_tree(positions)
        self.close_pairs = tree.query_ball_tree(tree, r=self.neighborhood_size)

        if num_obstacles > 0:
            self.tree_obstacles = kd_tree(self.obstacles[:num_obstacles])
            self.close_pairs_obstacles = tree.query_ball_tree(self.tree_obstacles, r=20)

        if num_attractors > 0:
            self.tree_attractors = kd_tree(self.attractors[:num_attractors])
            self.close_pairs_attractors = tree.query_ball_tree(self.tree_attractors, r=100)
            self.closest_pairs_attractors = tree.query_ball_tree(self.tree_attractors, r=50)

        for i in range(num_boids):
            self.close_pairs[i].remove(i)

        cohesion = self.cohesion_weight * self.get_cohesion()
        alignment = self.alignment_weight * self.get_alignment()
        separation = self.separation_weight * self.get_separation()

        if num_obstacles > 0:
            obstacles_repulsion = self.obstacles_repulsion_weight * self.get_obstacles_repulsion()

        if num_attractors > 0:
            attractors_cohesion = self.attractors_cohesion_weight * self.get_attractors_cohesion()
            attractors_repulsion = self.attractors_repulsion_weight * self.get_attractors_repulsion()

        self.velocities[:num_boids] += cohesion + alignment + separation

        if num_obstacles > 0:
            self.velocities[:num_boids] += obstacles_repulsion

        if num_attractors > 0:
            self.velocities[:num_boids] += attractors_cohesion
            self.velocities[:num_boids] += attractors_repulsion

        self.limit_velocity()
        self.angles = np.arctan2(self.velocities[:num_boids, 1], self.velocities[:num_boids, 0])

        self.update_colors()


    def add_boid(self, position):
        if self.num_boids == self.max_num_boids:
            print("Limit reached")
            return

        self.positions[self.num_boids] = position
        self.num_boids += 1


    def add_attractor(self, position):
        if self.num_attractors == self.max_num_attractors:
            print("Limit reached")
            return

        self.attractors[self.num_attractors] = position
        self.num_attractors += 1


    def add_obstacle(self, position):
        if self.num_obstacles == self.max_num_obstacles:
            print("Limit reached")
            return

        self.obstacles[self.num_obstacles] = position
        self.num_obstacles += 1
