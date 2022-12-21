import numpy as np
from statistics import mode
from kNN.kNN import kNN


class DROP3:

    def __init__(self, k=1, dist_metric='minkowski', r=2, weights="uniform"):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.weights = weights
        self.x_train = None
        self.y_train = None


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

            # frequencies = []
            # for z in classes:
            #     frequencies.append(classes.count(z))
            # frequency_check = all(ele == frequencies[0] for ele in frequencies)
            # if frequency_check:
            #     majority_class = None
            # else:
            majority_class = mode(classes)

            if label == majority_class:
                subset_x.append(x_train[i])
                subset_y.append(y_train[i])

        return np.array(subset_x), np.array(subset_y)

    def get_enemies(self, x_train, y_train, numeric_cols, nominal_cols):

        '''
        Sorting instances in S by distance to their nearest remaining "enemy",
        where an enemy is a neighbor of a different class. The final sorting should be
        from the longest distance to the shortest.
        '''

        kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
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
                    enemies[neighbors[j]] = neighbor_dist       # adding enemy (neighbor) and its distance
                    enemy_labels[neighbor_class] = neighbor_dist        # and its label

                # finding enemy at largest distance
            subset_enemies[x_train[i]] = sorted(enemies.items(), key=lambda x: x[1], reverse=True)[0]
            subset_labels[y_train[i]] = sorted(subset_enemies.items(), key=lambda x: x[1], reverse=True)[0]

        subset_enemies = sorted(subset_enemies.items(), key=lambda x: x[1], reverse=True)
        subset_labels = sorted(subset_labels.items(), key=lambda x: x[1], reverse=True)
        subset_x = sorted([key[1][1] for key in subset_enemies])
        subset_y = sorted([key[1][1] for key in subset_labels])

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
        other_associates = {}   # for the neighbor-ception later (neighbors of neighbors)
        other_aClasses = {}


            # iterating through instances P of S
        for i in range(len(S_points)):

                # k+1 nearest neighbors/associates of P
            kNN_config = kNN(k=self.k+1, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
            kNN_config.fit(S_points, S_labels, numeric_cols, nominal_cols)
            P_neighbors, P_labels, P_distances = kNN_config.get_neighbors(S_points[i])

            Pneighbors_classes = []
            P_associates = {}
            withP = []
            withoutP = []

                # iterating through neighbors of instance P
            for j in P_neighbors:

                Pneighbors_classes.append(P_labels[j])
                # kNN_neighbor = kNN_config.fit(P_neighbors, Pneighbors_classes, numeric_cols, nominal_cols)
                neighbors, labels, _ = kNN_config.get_neighbors(P_neighbors[j])
                Pneighbor_neighbors[P_neighbors[j]] = neighbors     # creating item in dict for this neighbor's own list of neighbors
                Pneighbor_nClasses[P_neighbors[j]] = P_labels[j]

                    # add P to each of its neighbors' lists of associates
                Pneighbor_associates[P_neighbors[j]] = S_points[i]
                Pneighbor_aClasses[P_neighbors[j]] = S_labels[i]

                for neighbor, nClass in zip(neighbors, range(len(Pneighbors_classes))):
                    neighbors_temp, labels_temp, distance_temp  = kNN_config.get_neighbors(neighbor)
                        # checking for association of this instance with its neighbors
                    index = [z for z in range(len(neighbors_temp)) if neighbors_temp[z] == P_neighbors[j]]
                    if P_neighbors[j] in neighbors_temp:
                        new_val = list(Pneighbor_associates[P_neighbors[j]])
                        new_val.append(neighbors_temp[index])
                            # replace dict value with new list including neighbor
                        Pneighbor_associates[P_neighbors[j]] = new_val


                        # if the instance has P as one of its nearest neighbors, then it is also an associate of P
                    if neighbor == S_points[i]:
                        P_associates[P_neighbors[j]] = P_neighbors[j], Pneighbors_classes[nClass]
                        Passociate = True

                if Passociate:

                        # testing for with
                    with_test = kNN_config.predict(P_neighbors[j])
                    if with_test == Pneighbors_classes[j]:      # if correctly classified with P as a neighbor
                        withP.append(with_test)

                        # testing for without
                    Ptest_points = T_points.copy()
                    Ptest_points.remove(S_points[i])
                    Ptest_labels = T_labels.copy()
                    Ptest_labels.remove(S_labels[i])
                    kNN_test = kNN_config.fit(Ptest_points, Ptest_labels, numeric_cols, nominal_cols)
                    without_test = kNN_test.predict(P_neighbors[j])
                    if without_test == Pneighbors_classes[j]:      # if correctly classified without P as a neighbor
                        withoutP.append(without_test)


                    if len(without_test) >= len(with_test):

                            # remove P from S
                        S_points.remove(S_points[i])
                        S_labels.remove(S_labels[i])

                            # remove P from its associate's list of nearest neighbors
                                # but NOT their list of associates
                        del Pneighbor_neighbors[P_neighbors[j]]
                        del Pneighbor_nClasses[P_neighbors[j]]

                            # since the associate must maintain k+1 nearest neighbors, find new neighbors so P is replaced
                        neighbors_new, labels_new, distance_new = kNN_test.get_neighbors(P_neighbors[j])
                        # and update dict item with its new neighbors and their classes
                        Pneighbor_neighbors[P_neighbors[j]] = neighbors_new
                        Pneighbor_nClasses[P_neighbors[j]] = labels_new

                            # finally, add this point to its new neighbor's list of associates
                        index = [z for z in range(len(neighbors_new)) if neighbors_new[z] not in neighbors]
                        other_associates[neighbors_new[index]] = P_neighbors[j]
                        other_aClasses[neighbors_new[index]] = Pneighbors_classes[j]

        return np.array(S_points), np.array(S_labels)

    def reduce_drop3(self, x_train, y_train, numeric_cols, nominal_cols):

        filtered_x, filtered_y = self.noise_filter(x_train, y_train, numeric_cols, nominal_cols)

        sorted_x, sorted_y = self.get_enemies(filtered_x, filtered_y, numeric_cols, nominal_cols)

        reduced_x, reduced_y = self.drop2(sorted_x, sorted_y, numeric_cols, nominal_cols)

        return np.array(reduced_x), np.array(reduced_y)












