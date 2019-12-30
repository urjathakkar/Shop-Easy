from __future__ import print_function
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from collections import deque, namedtuple
from flask import Flask, redirect, url_for, request
from flask import jsonify,json,render_template
import numpy
import pandas

dictgraph = { "Entry":(0,0),"101":(0,2),"201":(5,2),"301":(10,2),"401":(15,2),"501":(20,2),"610":(0,13),"710":(11,13),
             "102":(0,3),"103":(0,4),"104":(0,5),"105":(0,6),"106":(0,7),"107":(0,8),"108":(0,9),"109":(0,10),"110":(0,11),
             "202":(5,3),"203":(5,4),"204":(5,5),"205":(5,6),"206":(5,7),"207":(5,8),"208":(5,9),"209":(5,10),"210":(5,11),
             "302":(10,3),"303":(10,4),"304":(10,5),"305":(10,6),"306":(10,7),"307":(10,8),"308":(10,9),"309":(10,10),"310":(10,11),
             "402":(15,3),"403":(15,4),"404":(15,5),"405":(15,6),"406":(15,7),"407":(15,8),"408":(15,9),"409":(15,10),"410":(15,11),
             "502":(20,3),"503":(20,4),"504":(20,5),"505":(20,6),"506":(20,7),"507":(20,8),"508":(20,9),"509":(20,10),"510":(20,11),
             "609":(1,13),"608":(2,13),"607":(3,13),"606":(4,13),"605":(5,13),"604":(6,13),"603":(7,13),"602":(8,13),"601":(9,13),
             "709":(12,13),"708":(13,13),"707":(14,13),"706":(15,13),"705":(16,13),"704":(17,13),"703":(18,13),"702":(19,13),"701":(20,13)
}

#queue = ["Entry","304","101","209","110","403","109","301","310"]

def create_data_model(queue):
    data = {}
    itemCoordinates=[]
    for i in queue:
        itemCoordinates.append(dictgraph[i])
    data['locations'] = itemCoordinates
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def compute_euclidean_distance_matrix(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances

def main(queue):
    data = create_data_model(queue)

    manager = pywrapcp.RoutingIndexManager(len(data['locations']),data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'])

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30
    search_parameters.log_search = True

    assignment = routing.SolveWithParameters(search_parameters)

    if assignment:
        finalItem=[]
        #l=print_solution(manager, routing, assignment)
        listl=[]
        print('Objective: {}'.format(assignment.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            listl.append(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        listl.append(manager.IndexToNode(index))
        plan_output += 'Objective: {}\n'.format(route_distance)
        print(plan_output)
        for i in listl:
            finalItem.append(queue[i])
        print(finalItem)
    return finalItem


inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
  return Edge(start, end, cost)


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs


    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))
        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: inf for vertex in self.vertices}
        #print(distances)
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()
        while vertices:
            current_vertex = min(vertices, key=lambda vertex: distances[vertex])
            #print(current_vertex)
            vertices.remove(current_vertex)
            if distances[current_vertex] == inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                #print(alternative_route)
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex
        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path

        distance_between_nodes = 0
        for index in range(1, len(path)):
            for thing in self.edges:
                if thing.start == path[index - 1] and thing.end == path[index]:
                    distance_between_nodes += thing.cost

graph = Graph([("Entry", "101", 2),
    ("101", "102", 1), ("102", "103", 1), ("103", "104", 1), ("104", "105", 1), ("105", "106", 1), ("106", "107", 1), ("107", "108", 1), ("108", "109", 1), ("109", "110", 1),
    ("102", "101", 1), ("103", "102", 1), ("104", "103", 1), ("105", "104", 1), ("106", "105", 1), ("107", "106", 1), ("108", "107", 1), ("109", "108", 1), ("110", "109", 1),
    ("101", "201", 2), ("110", "210", 2), ("201", "202", 1), ("202", "203", 1), ("203", "204", 1), ("204", "205", 1), ("205", "206", 1), ("206", "207", 1), ("207", "208", 1), ("208", "209", 1), ("209", "210", 1),
    ("201", "101", 2), ("210", "110", 2), ("202", "201", 1), ("203", "202", 1), ("204", "203", 1), ("205", "204", 1), ("206", "205", 1), ("207", "206", 1), ("208", "207", 1), ("209", "208", 1), ("210", "209", 1),
    ("201", "301", 2), ("210", "310", 2), ("301", "302", 1), ("302", "303", 1), ("303", "304", 1), ("304", "305", 1), ("305", "306", 1), ("306", "307", 1), ("307", "308", 1), ("308", "309", 1), ("309", "310", 1),
    ("301", "201", 2), ("310", "210", 2), ("302", "301", 1), ("303", "302", 1), ("304", "303", 1), ("305", "304", 1), ("306", "305", 1), ("307", "306", 1), ("308", "307", 1), ("309", "308", 1), ("310", "309", 1),
    ("301", "401", 2), ("310", "410", 2), ("401", "402", 1), ("402", "403", 1), ("403", "404", 1), ("404", "405", 1), ("405", "406", 1), ("406", "407", 1), ("407", "408", 1), ("408", "409", 1), ("409", "410", 1),
    ("401", "301", 2), ("410", "310", 2), ("402", "401", 1), ("403", "402", 1), ("404", "403", 1), ("405", "404", 1), ("406", "405", 1), ("407", "406", 1), ("408", "407", 1), ("409", "408", 1), ("410", "409", 1),
    ("401", "501", 2), ("410", "510", 2), ("501", "502", 1), ("502", "503", 1), ("503", "504", 1), ("504", "505", 1), ("505", "506", 1), ("506", "507", 1), ("507", "508", 1), ("508", "509", 1), ("509", "510", 1),
    ("501", "401", 2), ("510", "410", 2), ("502", "501", 1), ("503", "502", 1), ("504", "503", 1), ("505", "504", 1), ("506", "505", 1), ("507", "506", 1), ("508", "507", 1), ("509", "508", 1), ("510", "509", 1),
    ("110", "610", 2), ("610", "609", 1), ("609", "608", 1), ("608", "607", 1), ("607", "606", 1), ("606", "605", 1), ("605", "604", 1), ("604", "603", 1), ("603", "602", 1), ("602", "601", 1), ("310", "601", 2),
    ("610", "110", 2), ("609", "610", 1), ("608", "609", 1), ("607", "608", 1), ("606", "607", 1), ("605", "606", 1), ("604", "605", 1), ("603", "604", 1), ("602", "603", 1), ("601", "602", 1), ("601", "310", 2),
    ("601", "710", 1), ("710", "709", 1), ("709", "708", 1), ("708", "707", 1), ("707", "706", 1), ("706", "705", 1), ("705", "704", 1), ("704", "703", 1), ("703", "702", 1), ("702", "701", 1), ("510", "701", 2),
    ("710", "601", 1), ("709", "710", 1), ("708", "709", 1), ("707", "708", 1), ("706", "707", 1), ("705", "706", 1), ("704", "705", 1), ("703", "704", 1), ("702", "703", 1), ("701", "702", 1), ("701", "510", 2)
    ])

app = Flask(__name__)

data=pd.read_csv("extra_coors.csv")
dataset=data.values
item_list=[]
shop_list=[]
queue=[]
map_coords_list=[]
def pointer(item):
    for i in range(147):
        value=dataset[i:i+1,0:1]
        if(value[0][0].lower()==item.lower()):
            ind=i
            map_c=dataset[i:i+1,2:3]
            c=dataset[i:i+1,1:2]
            shop_list.append(c[0][0])
            return map_c[0][0]
@app.route('/shop/')
def getStarted():
    print(request.method)
    return render_template("shop.html")

@app.route('/pointer')
def background_process():
    try:
        item = request.args.get('item', 0, type=str)
        item_list.append(item)
        item = pointer(item)
        item = item.split(',')
        newlist=[]
        for i in item:
            newlist.append(float(i))
        map_coords_list.append(newlist)
        return jsonify(result=newlist)
    except Exception as e:
        return str(e)

@app.route('/autocomplete')
def autocomplete():
    itemsFromDatabase = list(data.ITEMS)
    return jsonify(itemsFromDatabase=itemsFromDatabase)

def sortedList(list):
    sorted_item_list = []
    #print(item_list)
    queue.pop(0)
    #print(queue)
    for c in list[1:]:
        #rint("in if")
        ind = queue.index(c)
        #print(ind)
        sorted_item_list.append(item_list[ind])
    return sorted_item_list
@app.route('/pathShow')
def show_path():
    try:
        #print(shop_list)
        for c in shop_list:
            queue.append(str(c))
        queue.insert(0, "Entry")
        list = main(queue)
        list.pop()
        #print(list)
        new_item_list = sortedList(list)
        print(new_item_list)
        all_path = []
        temp_path=[]
        for i in range(len(list)):
            if(i<len(list)-1):
                temp_path = graph.dijkstra(list[i],list[i+1])
            else:
                break
            for c in temp_path:
                all_path.append(c)
            temp_path=[]
        # print(all_path)
        final_map_coords_list = []
        temp_list = []
        for c in all_path:
            if(c == "Entry"):
                final_map_coords_list.append([34.211138,-118.436288])
            else:
                for i in range(147):
                    value=dataset[i:i+1,1:2]
                    if(value[0][0] == int(c)):
                        map_c=dataset[i:i+1,2:3]
                        coordinates = map_c[0][0].split(',')
                        for i in coordinates:
                            temp_list.append(float(i))
                        final_map_coords_list.append(temp_list)
                        temp_list = []
        #print(final_map_coords_list)
        return jsonify(coordinates=final_map_coords_list, list=new_item_list)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
   app.run(debug=True)
