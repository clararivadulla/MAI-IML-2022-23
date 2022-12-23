import numpy as np
from statistics import mode
from kNN.kNN import kNN


class DROP3:

    def __init__(self, k=1, dist_metric='minkowski', r=2, weights="uniform", voting="uniform"):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.weights = weights
        self.x_train = None
        self.y_train = None
        self.voting = voting


    def noise_filter(self, x_train, y_train, numeric_cols, nominal_cols):   # ENN method

        '''
        1. Find kNN of each x
        2. Find the majority class of x's k neighbors
        3. If x's class doesn't match this, eliminate x
        '''

        kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
        kNN_config.fit(x_train, y_train, numeric_cols, nominal_cols)

        subset_x = []
        subset_y = []

        for i in range(len(x_train)):
            label = y_train[i]

            neighbors, labels, distance = kNN_config.get_neighbors(x_train[i])

            classes = []
            for j in range(0, self.k):
                neighbor_class = labels[j]
                classes.append(neighbor_class)

            try:
                majority_class = mode(classes)
            except:
                majority_class = classes[0]

            if label == majority_class:
                subset_x.append(x_train[i])
                subset_y.append(y_train[i])

#        print('end of noise filter, subset_x size is:', len(subset_x), np.shape(subset_x))

        return np.array(subset_x), np.array(subset_y)

    def get_enemies(self, x_train, y_train, numeric_cols, nominal_cols):

        '''
        Sorting instances in S by distance to their nearest remaining "enemy",
        where an enemy is a neighbor of a different class. The final sorting should be
        from the longest distance to the shortest.
        '''

        kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights, voting=self.voting)
        kNN_config.fit(x_train, y_train, numeric_cols, nominal_cols)

        subset_enemies = {}
        subset_labels = {}

        for i in range(len(x_train)):
            label = y_train[i]

            neighbors, labels, distance = kNN_config.get_neighbors(x_train[i])

            enemies = {}
            enemy_labels = {}

            for j in range(0, self.k):
                neighbor_class = labels[j]
                neighbor_dist = distance[j]

                if neighbor_class != label:
                    guilty_party = f'neighbors[{j}]'
                    enemies[guilty_party] = neighbor_dist, x_train[i]  # adding enemy (neighbor) and its distance
                    enemy_labels[neighbor_class] = neighbor_dist, y_train[i]  # and its label

            if len(enemies) == 0:
                increase = 1
                while len(enemies) == 0:
                    new_k = self.k + increase
                    kNN_config = kNN(k=new_k, dist_metric=self.dist_metric, r=self.r, weights=self.weights, voting=self.voting)
                    kNN_config.fit(x_train, y_train, numeric_cols, nominal_cols)
                    neighbors, labels, distance = kNN_config.get_neighbors(x_train[i])

                    for j in range(0, new_k):
                        neighbor_class = labels[j]
                        neighbor_dist = distance[j]

                        if neighbor_class != label:
                            guilty_party = f'neighbors[{j}]'
                            enemies[guilty_party] = neighbor_dist, x_train[i]  # adding enemy (neighbor) and its distance
                            enemy_labels[neighbor_class] = neighbor_dist, y_train[i]  # and its label

                    increase += 1

                # selecting enemy at farthest distance
            subset_enemies[f'x_train[{i}]'] = sorted(enemies.values(), key=lambda x: x[0], reverse=True)[0]
            subset_labels[f'y_train[{i}]'] = sorted(enemy_labels.values(), key=lambda x: x[0], reverse=True)[0]

        subset_enemies = sorted(subset_enemies.values(), key=lambda x: x[0], reverse=True)
        subset_labels = sorted(subset_labels.values(), key=lambda x: x[0], reverse=True)
        subset_x = [value[1] for value in subset_enemies]
        subset_y = [value[1] for value in subset_labels]
        
        # print(f'final check, subset_x and subset_y, sorted =\n{subset_x}\n{subset_y}')

        return np.array(subset_x), np.array(subset_y)

    def drop2(self, x_train, y_train, numeric_cols, nominal_cols):

        '''
        1. S = T    (subset = original)
        2. For each P in S:
            - Find k+1 nearest neighbors of P
            - Add P to each neighbor's list of associates
        3. For each P in T:
            - with: # associates correctly classified with P as a neighbor
            - without: # associates correctly classified without P as a neighbor
                 * Where an associate is an instance that has P as one of its k nearest neighbors
            If without >= with:
                - Remove P from S
                For each associate A of P:
                    - Remove P from A's list of nearest neighbors
                    - Find new nearest neighbor to replace P
                    - Add A to new neighbor's list of associates
        '''

        T_points = x_train
        T_labels = y_train

        S_points = T_points.copy()
        S_labels = T_labels.copy()

        Pneighbor_associates = {}
        Pneighbor_aClasses = {}
        Pneighbor_neighbors = {}
        Pneighbor_nClasses = {}
        Passociate_neighbors = {}
        Passociate_nClasses = {}
        other_associates = {}   # for the associate-ception later (associates of neighbors of neighbors)
        other_aClasses = {}


            # iterating through instances P of S, where the instance in question, "P", is S_points[i]
        for i in range(len(S_points)):

                # as values are deleted in S_points, need to ensure the loop still iterates through all of its points
            corr_trans = len(T_points) - len(S_points)
            Spoints_index = i - corr_trans
            Slabels_index = i - corr_trans

                # k+1 nearest neighbors of P
            kNN_config = kNN(k=self.k+1, dist_metric=self.dist_metric, r=self.r, weights=self.weights, voting=self.voting)
            kNN_config.fit(S_points, S_labels, numeric_cols, nominal_cols)
            P_neighbors, P_labels, P_distances = kNN_config.get_neighbors(S_points[Spoints_index])

            P_associates = {}
            withP = 0
            withoutP = 0

                # iterating through neighbors of instance P
            for j in range(len(P_neighbors)):
                Passociate = False

                    # getting this neighbor's own list of neighbors
                neighbors, labels, _ = kNN_config.get_neighbors(P_neighbors[j])
                Pneighbor_neighbors[f'P_neighbors[{j}] of S_points[{Spoints_index}]'] = neighbors
                Pneighbor_nClasses[f'P_labels[{j}] of S_labels[{Slabels_index}]'] = labels

                    # add P to each of its neighbors' lists of associates
                Pneighbor_associates[f'P_neighbors[{j}] of S_points[{Spoints_index}]'] = S_points[Spoints_index]
                Pneighbor_aClasses[f'P_labels[{j}] of S_labels[{Slabels_index}]'] = S_labels[Slabels_index]

                # checking for association of the instance "P_neighbors[j]" with its own neighbors, "neighbors"
                    # by seeing if, when we look for those neighbors' neighbors, P_neighbors[j] is one of them
                for neighbor, nClass in zip(neighbors, range(len(labels))):
                    neighbors_temp, labels_temp, distance_temp = kNN_config.get_neighbors(neighbor)
                    indices = [z for z in range(len(neighbors_temp)) if np.array_equal(neighbors_temp[z], P_neighbors[j])]
                    if P_neighbors[j] in neighbors_temp:
                        new_val = list(Pneighbor_associates[f'P_neighbors[{j}] of S_points[{Spoints_index}]'])
                        # new_lab = list(Pneighbor_aClasses[f'P_labels[{j}] of run[{i}]'])
                        for index in indices:
                            new_val.append(neighbors_temp[index])
                            # new_lab.append(labels_temp[index])
                        # replace dict value with new list including neighbor
                        Pneighbor_associates[f'P_neighbors[{j}] of run[{i}]'] = new_val
                        # Pneighbor_aClasses[f'P_labels[{j}] of run[{i}]'] = new_lab
                    # checking if the neighbor of P's neighbor is also an associate of P
                    indices = [z for z in range(len(neighbors_temp)) if np.array_equal(neighbors_temp[z], S_points[Spoints_index])]
                    if S_points[Spoints_index] in neighbors_temp:
                        for index in indices:
                            P_associates[f'neighbors_temp[{index}] of neighbor[{nClass}]'] = neighbors_temp[index], labels_temp[index]
                            neighbors_extratemp, labels_extratemp, _ = kNN_config.get_neighbors(P_neighbors[j])
                            Passociate_neighbors[f'neighbors_temp[{index}] of neighbor[{nClass}] of S_points[{Spoints_index}]'] = neighbors_extratemp
                            Passociate_nClasses[f'neighbors_temp[{index}] of neighbor[{nClass}] of S_points[{Spoints_index}]'] = labels_extratemp
                            Passociate = True

                    # checking if the instance "P_neighbors[j]" is itself an associate of P
                        # ie if P is one of its k nearest neighbors
                if S_points[Spoints_index] in neighbors:
                    P_associates[f'P_neighbors[{j}]'] = P_neighbors[j], P_labels[j]
                    Passociate_neighbors[f'P_neighbors[{j}] of S_points[{Slabels_index}]'] = neighbors
                    Passociate_nClasses[f'P_labels[{j}] of S_labels[{Slabels_index}]'] = labels
                    Passociate = True

                # after having gone through its neighbors associations and whatnot,
                    # check how important P is for classification of its associates
            if Passociate:
                Passociate_points = [value[0] for value in P_associates.values()]
                Passociate_labels = [value[1] for value in P_associates.values()]

                for associate, aLabel in zip(Passociate_points, Passociate_labels):
                        # testing for with P
                    with_test = kNN_config.predict(associate)
                        # if correctly classified with P as a neighbor, then append to "with" list
                    if with_test == aLabel:
                        withP += 1

                        # testing for without P
                    Ptest_points = T_points.copy()
                    Ptest_labels = T_labels.copy()
                    Ptest_index = int(np.argwhere(Ptest_points == S_points[Spoints_index])[0][0])
                    Ptest_points = np.delete(Ptest_points, Ptest_index, 0)
                    Ptest_labels = np.delete(Ptest_labels, Ptest_index, 0)
                    kNN_config.fit(Ptest_points, Ptest_labels, numeric_cols, nominal_cols)
                    without_test = kNN_config.predict(associate)
                        # if correctly classified without P as a neighbor, then append to "without" list
                    if without_test == aLabel:
                        withoutP += 1

                # print('checking tests, withoutP =', withoutP, 'and withP =', withP)

                if (withoutP >= withP) and (withoutP != 0):
                # if (withoutP > withP) and (withoutP != 0):
                        # remove P from S
                    removed_point = S_points[Spoints_index]
                    removed_label = S_labels[Slabels_index]
                    S_points = np.delete(S_points, Spoints_index, 0)
                    S_labels = np.delete(S_labels, Slabels_index, 0)

                        # remove P from its associates' lists of nearest neighbors
                            # but NOT from the associates' own lists of associates
                    for aitem, aclass in zip(Passociate_neighbors.keys(), Passociate_nClasses.keys()):
                        updated = list(Passociate_neighbors[aitem])
                        updated = [neigh for neigh in updated if not (neigh == removed_point).all()]
                        Passociate_neighbors[aitem] = updated

                        updated2 = list(Passociate_nClasses[aclass])
                        updated2 = [labe for labe in updated2 if not (labe == removed_label).all()]
                        Passociate_nClasses[aclass] = updated2

                            # since the associate must maintain k+1 nearest neighbors, find new neighbors so P is replaced
                        kNN_config.fit(S_points, S_labels, numeric_cols, nominal_cols)
                        neighbors_updated, labels_updated, distance_updated = kNN_config.get_neighbors(Passociate_neighbors[aitem])
                            # and update dicts with the associate's new neighbors and their classes
                        Passociate_neighbors[aitem] = neighbors_updated
                        Passociate_nClasses[aclass] = labels_updated

                        # finally, add this point to the associate's new neighbor's list of associates  (enter: associate-ception)
                            # looking for the new neighbor
                        index = [z for z in range(len(neighbors_updated)) if neighbors_updated[z] not in neighbors]
                        new_neighbor = neighbors_updated[index]
                        other_associates[f'Passociate_neighbors[{aitem}] of run[{i}]'] = Passociate_neighbors[aitem]
                        other_aClasses[f'Passociate_nClasses[{aclass}] of run[{i}]'] = Passociate_nClasses[aclass]

                # print('currently, S_points size is:', len(S_points), np.shape(S_points))

        return np.array(S_points), np.array(S_labels)

    def reduce_drop3(self, x_train, y_train, numeric_cols, nominal_cols):

        filtered_x, filtered_y = self.noise_filter(x_train, y_train, numeric_cols, nominal_cols)

        sorted_x, sorted_y = self.get_enemies(filtered_x, filtered_y, numeric_cols, nominal_cols)

        reduced_x, reduced_y = self.drop2(sorted_x, sorted_y, numeric_cols, nominal_cols)

        return np.array(reduced_x), np.array(reduced_y)




