import trimesh
import numpy as np
import argparse

import os

B_MIN = np.array([-0.5,-0.5,-0.5])
B_MAX = np.array([0.5,0.5,0.5])
def parse_args():
    parser = argparse.ArgumentParser(description='Train AMRSR')
    # data setting
    parser.add_argument('--root_dir', type=str, default='/vol/vssp/cvpnobackup_orig/scratch_4weeks/mp01223/datasets/CUFED')
    parser.add_argument('--render_dir', type=str, default='/vol/vssp/cvpnobackup_orig/scratch_4weeks/mp01223/datasets/CUFED')
    # train setting
    
    return parser.parse_args()



def select_sampling_method(subjects,obj_dir):
    total=[]
    c1=[]
    c2=[]
    for subject in subjects:
        name_HR=subject+'_HR.obj'
        name_LR=subject+'_LR.obj'
        #print("name_mesh",name_LR)
        ##print(self.mesh_dic)
        mesh_HR_name= obj_dir+"/"+name_HR
        mesh_LR_name= obj_dir+"/"+name_LR
        #print(mesh_HR_name,mesh_LR_name)
        mesh_LR=trimesh.load(mesh_LR_name)
        mesh_HR=trimesh.load(mesh_HR_name)
        surface_points, _ = trimesh.sample.sample_surface(mesh_HR, 4 * 6000) #20000 points sampled on the meshes
        sample_points = surface_points + np.random.normal(scale=0.06, size=surface_points.shape)

        #surface_points_LR, _ = trimesh.sample.sample_surface(mesh_LR, 4 * self.num_sample_inout) #20000 points sampled on the meshes
        #sample_points_LR = surface_points_LR + np.random.normal(scale=self.opt.sigma, size=surface_points_LR.shape)

        # add random points within image space
        length = B_MAX - B_MIN
        random_points = np.random.rand(6000// 4, 3) * length + B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)
        ####print("ok1")
        inside_HR = mesh_HR.contains(sample_points) #non mi lascia
        ####print("ok2")
        inside_points_HR = sample_points[inside_HR]
        outside_points_HR = sample_points[np.logical_not(inside_HR)]

        
        #sample_points_LR = np.concatenate([sample_points_LR, random_points], 0)
        #np.random.shuffle(sample_points_LR)

        #sample_points_LR = np.concatenate([sample_points_LR, random_points], 0)
        inside_LR = mesh_LR.contains(sample_points)
        inside_points_LR = sample_points[inside_LR]
        outside_points_LR = sample_points[np.logical_not(inside_LR)]

        #inside points lr in hr
        #print(inside_points_LR[0])
        

        
        #print(inside_points_LR.shape)
        #trovo la differenza tra inside point LR e inside point HR
        #e' un insieme di punti che da una parte sono inside mentre dall'altra outside o viceversa
        #come posso usarlo???
        # se sono inside in hr e outside in lr 1
        #se sono outside in hr e inside in lr 0
        #se sono uguali? prendo i punti inside e controllo in inside_points_HR e outside_points_HR, se inside points hr 1 se no 0. creo la ground truth cosi#

        ####print("ok3")
        nin_LR = inside_points_LR.shape[0]
        inside_points_LR = inside_points_LR[
                    :6000// 2] if nin_LR > 6000 // 2 else inside_points_LR
        outside_points_LR = outside_points_LR[
                        :6000 // 2] if nin_LR > 6000 // 2 else outside_points_LR[:(6000 - nin_LR)]
        nin_HR = inside_points_HR.shape[0]
        #print(inside_points_LR[0])
        inside_points_HR_new = inside_points_HR[
                    :6000 // 2] if nin_HR > 6000 // 2 else inside_points_HR
        outside_points_HR_new = outside_points_HR[
                        :6000 // 2] if nin_HR > 6000 // 2 else outside_points_HR[:(6000 - nin_HR)]
        ####print("ok5")    
        #
        label_disp_inside=np.ones((1,6000//2))
        label_disp_outside=np.zeros((1,6000//2))
  #         
        count1=0
        count2=0
        #print(len(list1))
        for i in range(inside_points_LR.shape[0]):
        #count1+=1
            if inside_points_LR[i] in outside_points_HR:
                label_disp_inside[0][i]=0
                count1+=1
            #
            if outside_points_LR[i] in inside_points_HR:
                label_disp_outside[0][i]=1
                count2+=1
            #
        tot=count1+count2
        c1.append(count1)
        c2.append(count2)
        total.append(tot)   
        #print(np.array_equal(inside_points_LR,inside_points_HR))
        #print(count3,count4)

        #print("ok")

                #print(i)
        # count=0        
        # for i in inside_points_LR:
        #     for el in outside_points_HR:
        #         if np.array_equal(i,el):
        #             count+=1
        # print(count1,count)
        #print(count,count2)   

    return total,c1,c2


if __name__ == '__main__':
    opt = parse_args()
    subject=os.listdir(opt.render_dir)
   
   
    tot,c1,c2 = select_sampling_method(subject,opt.root_dir)

    tot_fin=np.array(tot)
    c1_fin=np.array(c1)
    c2_fin=np.array(c2)
    print(np.average(tot_fin),np.average(c1),np.average(c2))