from __future__ import division,print_function
import clickpoints
import glob


def Create_DB(Name,pic_path,db_path,pic_pos):
    db = clickpoints.DataFile(Name, 'w')
    images = glob.glob(pic_path)
    print(len(images))
    # def setImage(self, filename=None, path=None, frame=None, external_id=None, timestamp=None, width=None, height=None,
    #              id=None, layer=None):
    layer_dict = {"MinP": 0, "MinIndices": 1, "MaxP":2, "MaxIndices": 3}
    db.setPath(db_path, 1)
    for image in images:
        path = "/".join(image.split("/")[:-1])
        file = image.split("/")[-1]
        idx = file.split("_")[2][3:]
        layer_str = file.split("_")[-1][1:-4]
        if not file.count(pic_pos):
            continue
        if layer_str.count("MinProj"):
            layer = 0
        elif layer_str.count("MinIndices"):
            layer = 1
        elif layer_str.count("MaxProj"):
            layer = 2
        elif layer_str.count("MaxIndices"):
            layer = 3
        else:
            raise ValueError("No known layer!")
        print(idx, layer)
        image = db.setImage(filename=file, path=1, layer=layer)#, frame=int(idx))
        image.sort_index = int(idx)
        image.save()
    # db.db.close()

# path_list = glob.glob('/mnt/cmark2/T-Cell-Motility/2017-07-18/*/*/*/*/*')
# path_list = [path for path in path_list if not path.count("video")]
# print(path_list)
# for i in path_list:
#     Name = "layers_"+"_".join(i.split("/")[-5:])+".cdb"
#     Name = "_".join(Name.split("-"))
#     # Name = "/".join(i.split("/")[:5]) + "/" + Name
#     Name = '/home/user/TCell/2017-07-18/' + Name
#     Fin_Name = '/home/user/TCell/2017-08-29/layers_2017_08_29_'
#     if Name.count('1hnachMACS'):
#         Fin_Name += '1h_'
#     elif Name.count('24hnachMACS'):
#         Fin_Name +='24h_'
#     if Name.count('1himgel'):
#         Fin_Name += '1hiG_'
#     elif Name.count('24himgel'):
#         Fin_Name += '24hiG_'
#     if Name.count('Colitisulcerosa'):
#         Fin_Name += 'CU_'
#     elif Name.count('Kontrolle'):
#         Fin_Name += 'Kon_'
#     elif Name.count('MorbusCrohn'):
#         Fin_Name += 'MC_'
#     Fin_Name += i.split('/')[-1] + '_'
#     if Name.count('1.2Gel'):
#         Fin_Name += '12Gel.cdb'
#     elif Name.count('2.4Gel'):
#         Fin_Name += '24Gel.cdb'
#     pic_path = i + '/*.tif'
#     db_path = i + '/'
#     pos = "pos0"
#     print(Fin_Name,Name,pic_path,db_path)
#     # Create_DB(Name, pic_path, db_path, pos)

path_list = glob.glob("/media/user/Elements/3D/*")
path_list = [path for path in path_list if path.count("TAG14")]
path_list = [path for path in path_list if path.count("_migration_")]
path_list = [path for path in path_list if path.count("_aufgetaut_") or path.count("_frisch_")]
path_list = [path for path in path_list if not path.count("20160613_NK cells_migration_frisch_TAG14_171")]
pic_pos_list = ["x0_y0","x0_y1","x1_y0","x1_y1"]
for i in path_list:
    for j in pic_pos_list:
        Name = "/home/user/NK Test/NK_Tracks/" + i.split("/")[5] + "_" + j + ".cdb"
        pic_path = i + "/overnight/*.tif"
        db_path = i + "/overnight/"
        # print (Name,pic_path,db_path)
        Create_DB(Name,pic_path, db_path, j)
print("-----Done-----")