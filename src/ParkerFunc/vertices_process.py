import pymesh
import numpy as np
from absl import app, flags
import os
def mean_obj_add_delta():
    mean_bird_mesh = pymesh.load_mesh("cachedir/template_shape/BirdReOrined_Unwrap_Half_cut_by_body.obj")
    delta_v_data=open("vertices_output/predict_vertices.txt","r")
    lines=delta_v_data.readlines()
    count=0
    vertices=mean_bird_mesh.vertices
    # print(lines[0])
    # exit()
    new_vetices=np.zeros((vertices.shape[0],vertices.shape[1]),dtype=float)
    for index,line in enumerate(lines):
        delta_v=line.split()
        # print(delta_v)
        # exit()
        delta_v_x=float(delta_v[0])
        delta_v_y=float(delta_v[1])
        delta_v_z=float(delta_v[2])
        new_vetices[index][0]=delta_v_x
        new_vetices[index][1]=delta_v_y
        new_vetices[index][2]=delta_v_z
    new_mesh=pymesh.form_mesh(new_vetices,mean_bird_mesh.faces)
    pymesh.save_mesh("vertices_output/mesh_edited.obj",new_mesh)
def read_obj_and_edited_vertices():
    obj_file=open("cachedir/template_shape/BirdReOrined_Unwrap_Half_cut_by_body.obj","r")
    new_obj_file=open("vertices_output/edited_mesh.obj")
    lines=obj_file.readlines
    for line in lines:
        vertex=line.split()
        if(vertex[0]=="v"):
            x=float(vertex[1])
            y=float(vertex[2])
            z=float(vertex[3])
            new_obj_file.write("v "+x+" "+y+" "+z+"\n")
        else:
            new_obj_file.write(line)

    new_obj_file.close()

    obj_file.close()
def write_obj(obj_file,vertices_be_project):
    new_obj_file = open("vertices_output/edited_mesh.obj", "w")
    lines = obj_file.readlines()
    # print(lines)
    # exit()
    vert_count = 0
    for line in lines:
        vertex = line.split()
        if (vertex[0] == "v"):
            x = vertices_be_project[0][vert_count][0].item()
            y = vertices_be_project[0][vert_count][1].item()
            z = vertices_be_project[0][vert_count][2].item()
            new_obj_file.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
            vert_count += 1
        else:
            new_obj_file.write(line)

    new_obj_file.close()

    obj_file.close()

def generate_edited_obj(shape_path,vertices_be_project,dir=None,idx=None):
    obj_file = open(shape_path, "r")
    if(dir==None and idx==None):
        new_obj_file = open("vertices_output/edited_mesh.obj", "w")
    else:
        # directory=str(name)
        # parent_dir="/home/parker/ucmr_v1/output_obj"
        # path=os.path.join(parent_dir, directory)
        # os.mkdir(path)
        # print("create directory "+str(name))
        #
        obj_path=dir+idx+".obj"
        new_obj_file=open(obj_path,"w")
    lines = obj_file.readlines()
    # print(lines)
    # exit()
    vert_count = 0
    for line in lines:
        vertex = line.split()
        if (vertex[0] == "v"):
            x = vertices_be_project[0][vert_count][0].item()
            y = vertices_be_project[0][vert_count][1].item()
            z = vertices_be_project[0][vert_count][2].item()
            new_obj_file.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
            vert_count += 1
        else:
            new_obj_file.write(line)

    new_obj_file.close()

    obj_file.close()
    return 0
def main(_):
    mean_obj_add_delta()
    print("done")
    return 0
if __name__ == '__main__':
    app.run(main)
