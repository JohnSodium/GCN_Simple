import os
import pandas as pd
import count_lc
import GCN
import CNN


prev_frame = None
node_data = None
vehicle_count = 0
spatial_features = []
original_vehicles = []

###### load data ######
lc_frame = count_lc.get_Swap_Frame()    # Frames where lane changes occur

for i in range(0, 1):

    source_features = []
    target_features = []
    node_features = []

    new_path = f'./Graph/{lc_frame[i]}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    feature_vector = pd.read_csv(f"./new_input_feature_matrix_5m10m/"
                                 f"{lc_frame[i]}_features_with_label.csv")

    for j in range(len(feature_vector) - 1, -1, -1):
        cur_frame = int(feature_vector["Frame"][j])

        if cur_frame != prev_frame:

            temp_x = []
            temp_y = []
            x = []
            y = []
            vehicle_list = []
            edges = []
            adjacency = []

            node_data = pd.read_csv(f"./adjacency_5m10m/{cur_frame}_output.csv")

            # Find connections between vehicles using the adjacency matrix
            for index, row in node_data.iterrows():
                connection = []
                for k in range(1, len(row.index)):
                    if k == 1 and int(row[0]) not in vehicle_list:  # only add vehicle_id once
                        vehicle_list.append(int(row[0]))
                    # Create an edge when there is a connection between two vehicles
                    if row[k] == 1:
                        # print(int(row[0]), int(row.index[k]))
                        edges.append((int(row[0]), int(row.index[k])))

                    # Create adjacency matrix that we can manipulate
                    connection.append((row[k]))
                adjacency.append(connection)

            if vehicle_count == 0:
                vehicle_count = len(vehicle_list)

            vehicle_list = sorted(vehicle_list)

            if j == len(feature_vector) - 1:
                original_vehicles = vehicle_list

            for k in range(len(vehicle_list)):
                separate_by_id = feature_vector[feature_vector['Vehicle_ID'] == vehicle_list[k]]
                separate_by_frame = separate_by_id[separate_by_id['Frame'] == cur_frame]

                temp_x.append(separate_by_frame['Local_X'].values)
                temp_y.append(separate_by_frame['Local_Y'].values)

            for k in temp_x:
                x.append(float(k))

            for k in temp_y:
                y.append(float(k))

            # print(original_vehicles)
            # print(vehicle_list)

            originalptr = 0
            curptr = 0
            for k in range(len(original_vehicles)):
                # print(originalptr, curptr, vehicle_list, original_vehicles)
                if original_vehicles[originalptr] == vehicle_list[curptr]:
                    originalptr += 1
                    curptr += 1
                elif original_vehicles[originalptr] != vehicle_list[curptr]:
                    # print('missing vehicle')
                    vehicle_list.insert(curptr, original_vehicles[originalptr])
                    x.insert(curptr, 0)
                    y.insert(curptr, 0)
                    adjacency.insert(curptr, [0 for i in range(len(vehicle_list))])
                    for l in range(len(vehicle_list)):
                        if len(adjacency[l]) != len(vehicle_list):
                            adjacency[l].insert(curptr, 0)

                    originalptr += 1
                    curptr += 1

            # Self Loop
            for k in range(len(adjacency)):
                adjacency[k][k] = 1

            result = GCN.setup(cur_frame, vehicle_list, x, y, adjacency)
            spatial_features.append(result)

            prev_frame = cur_frame
