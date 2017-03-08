from __future__ import division, print_function
import clickpoints
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

db = clickpoints.DataFile("./241_tracked.cdb")

GT_Type = db.getMarkerType(name="GT")
GT_Tracks = db.getTracks(type=GT_Type)
GT_Data = {}
for t in GT_Tracks:
    GT_Data.update({t.id: {}})
    for m in db.getMarkers(type=GT_Type, track=t):
        GT_Data[t.id].update({m.image_id: [m.x, m.y]})

Auto_Type = db.getMarkerType(name="PT_Track_Marker")
Auto_Pred_Type = db.getMarkerType(name="PT_Prediction_Marker")
Auto_Tracks = db.getTracks(type=Auto_Type)
Auto_Data = {}
for track in Auto_Tracks:
    Auto_Data.update({track.id: {}})
    times = [m.image_id for m in db.getMarkers(type=Auto_Type, track=track)]
    if times == []:
        continue
    max_time = np.amax(times)
    min_time = np.amin(times)
    for t in range(min_time, max_time):
        if t in times:
            m = db.getMarkers(image=t, track=track, type=Auto_Type)[0]
            Auto_Data[track.id].update({m.image_id: [m.x, m.y, float(m.text.split("Prob")[1])]})
        else:
            try:
                m = db.getMarkers(image=t, type=Auto_Pred_Type, text="Track %s"%track.id)[0]
            except IndexError:
                print(t, track.id)
                raise
            Auto_Data[track.id].update({m.image_id: [m.x, m.y, 0.]})

    # for m in db.getMarkers(type=Auto_Type, track=t):
    #     try:
    #         Auto_Data[t.id].update({m.image_id: [m.x, m.y, float(m.text.split("Prob")[1])]})
    #     except AttributeError:
    #         pass
Auto_Data.pop(0)

all_probs = []
[all_probs.extend(a.values()) for a in Auto_Data.values()]
all_probs = np.asarray(all_probs).T[2]
mean_probs = [np.mean([i[2] for i in a.values() if len(i)>0]) for a in Auto_Data.values() if len(a.values()) > 3]

plt.ylabel("Normed frequency")
plt.xlabel("Log-Likelihood")
plt.hist(all_probs, bins=200, normed=True, alpha=1, label="Probabilities of Detections")
plt.hist(mean_probs, bins=200, normed=True, alpha=0.5, label="Mean Probabilities of Tracks longer than 3")
plt.legend()
plt.savefig("Probs.pdf")
plt.show()

plt.figure()
plt.ylabel("Normed frequency")
plt.xlabel("Length of Track")
all_lengths, length_weights = np.asarray([[len(a.values()),
                                           np.mean([i[2] for i in a.values() if len(i) > 0])] for a in Auto_Data.values()]).T
plt.hist(all_lengths, bins=np.amax(all_lengths), normed=True, label="Lengths of Tracks")
plt.hist(all_lengths, bins=np.amax(all_lengths), normed=True, weights=length_weights-np.amin(length_weights), alpha=0.5, label="Weighted Lengths of Tracks")
plt.legend()
plt.savefig("Lengths_1D.pdf")
plt.show()


plt.figure()
plt.xlabel("Length of Track")
plt.ylabel("Mean Log-Likelihood of Track")
all_lengths, length_weights = np.asarray([[len(a.values()),
                                           np.mean(np.array(a.values(), ndmin=2).T[2], axis=0)] for a in Auto_Data.values() if len(a.values())>3]).T
hist = plt.hist2d(all_lengths, length_weights, bins=50, label="Probabilities and Tracks (longer than 3)", cmap="coolwarm", normed=True)
cbar = plt.colorbar()
cbar.set_label("Normed frequency", rotation=90)
plt.savefig("Lengths_2D.pdf")
plt.show()

# mean_dist = {}
# match = {}
# min_points = {}
# min_match = {}
# for g in GT_Data:
#     gt_track = GT_Data[g]
#     m = []
#     # min_points.update({g: [Auto_Data.keys()[
#     #                            np.nanargmin([np.linalg.norm(np.array(gt_track[t]) - np.array(Auto_Data[a][t][:2]))
#     #                                          if t in Auto_Data[a].keys() else np.nan for a in Auto_Data])]
#     #                        for t in gt_track if (np.any([Auto_Data[a].keys().count(t) for a in Auto_Data])
#     #                                              and np.any(
#     #         [np.linalg.norm(np.array(gt_track[t]) - np.array(Auto_Data[a][t][:2])) < 14
#     #          for a in Auto_Data if t in Auto_Data[a].keys()]))]})
#
#     mins = []
#     for t in gt_track:
#         if (np.any([Auto_Data[a].keys().count(t) for a in Auto_Data])
#             and np.any([np.linalg.norm(np.array(gt_track[t]) - np.array(Auto_Data[a][t][:2])) < 50
#                         for a in Auto_Data if t in Auto_Data[a]])):
#             mins.append(Auto_Data.keys()[
#                             np.nanargmin([np.linalg.norm(np.array(gt_track[t]) - np.array(Auto_Data[a][t][:2]))
#                                           if t in Auto_Data[a] else np.nan for a in Auto_Data])])
#     min_points.update({g: mins})
#     min_set = set(min_points[g])
#     min_match.update({g: [[min_points[g].count(a), a] for a in Auto_Data]})
#     min_match[g].sort(reverse=True)
#     for a in Auto_Data:
#         auto_track = Auto_Data[a]
#         # if min(auto_track.keys()) > max(gt_track.keys()):
#         if set(auto_track.keys()).isdisjoint(set(gt_track.keys())):
#             pass
#         else:
#             times = set(auto_track.keys()).intersection(
#                 gt_track.keys())  # range(min(auto_track.keys()), max(auto_track.keys()))
#             try:
#                 m.append([a, np.mean(
#                     [np.linalg.norm(np.array(gt_track[t][:2]) - np.array(auto_track[t][:2])) for t in times]), np.sum(
#                     [np.linalg.norm(np.array(gt_track[t][:2]) - np.array(auto_track[t][:2])) for t in times]),
#                           max(times) - min(times)])
#             except ValueError:
#                 print(times)
#                 raise
#     m = np.array(m)
#     arg = m.T[3].argsort()[::-1]
#     m = m[arg]
#     mean_dist.update({g: m})
#     match.update({g: []})
#     for mm in min_match[g]:
#         times = Auto_Data[mm[1]].keys()
#         for i in match[g]:
#             if set(times).isdisjoint(set(Auto_Data[i].keys())):
#                 pass
#             else:
#                 break
#         else:
#             match[g].append(mm[1])
#             # for dat in m:
#             #     if dat[1] < 200:
#             #         a = int(dat[0])
#             #         times = Auto_Data[a].keys()
#             #         for i in match[g]:
#             #             j = [l for l, k in enumerate(m) if k[0] == i]
#             #             if set(times).isdisjoint(set(Auto_Data[i].keys())) and dat[2] < m[j, 2]:
#             #                 pass
#             #             else:
#             #                 break
#             #         else:
#             #             match[g].append(a)

min = {}
for g in GT_Data:
    min.update({g:{}})
    for t in GT_Data[g]:
        if np.any([t in Auto_Data[a] for a in Auto_Data]):
            norms = np.linalg.norm(
                    np.asarray([Auto_Data[a][t][:2] for a in Auto_Data if t in Auto_Data[a]]
                           )-GT_Data[g][t], axis=1)
            idx = np.argmin(norms)
            if norms[idx] < 20:
                min[g].update({t: [a for a in Auto_Data if t in Auto_Data[a]][idx]})
match = {}
for g in min:
    match.update({g: {}})
    times = [t for t in min[g]]
    hits = {}
    for t in times:
        hits.update({min[g][t]: min[g].values().count(min[g][t])})
    while True:
        try:
            hit_id = [h for h in hits][np.argmax([hits[h] for h in hits])]
        except ValueError:
            print(hits, g)
            raise
        hit_n = hits.pop(hit_id)
        print(g,[set(Auto_Data[i].keys()).isdisjoint(set(Auto_Data[hit_id].keys())) for i in match[g]])
        if np.all([set(Auto_Data[i].keys()).isdisjoint(set(Auto_Data[hit_id].keys())) for i in match[g]]):
            match[g].update({hit_id: hit_n})
        if hits == {}:
            break

print("setting markers")
match_type = db.getMarkerType(name="Matches")
if match_type is None:
    match_type = db.setMarkerType(name="Matches", color="#FFFFFF", mode=db.TYPE_Track)
else:
    db.deleteTracks(type=match_type)
    db.deleteMarkers(type=match_type)

for g in match:
    track = db.setTrack(type=match_type, id=(10000 + g))
    print("%s" % g)
    for a in match[g]:
        print("\t%s" % a)
        for t in Auto_Data[a]:
        # for marker in db.getMarkers(track=i):
            print("\t\t%s" % t)
            db.setMarker(image=t, x=Auto_Data[a][t][0], y=Auto_Data[a][t][1], track=track.id, text="%s" % g)

print("Evaluating matches")
com={}
alex={}
eval = {}
for g in GT_Tracks:
    eval.update({g.id: {}})
    auto = {}
    for m in db.getMarkers(track=(g.id + 10000)):
        auto.update({m.image_id: [m.x, m.y]})
    com.update({g.id: auto})
    gt = {}
    for m in db.getMarkers(type=GT_Type, track=g):
        gt.update({m.image_id: [m.x, m.y]})
    for t in range(2, np.amax(gt.keys())+1):
        if t-1 in gt and t in gt:
            gt[t].append(gt[t][0]-gt[t-1][0])
            gt[t].append(gt[t][1]-gt[t-1][1])
        elif t in gt:
            gt[t].append((gt[t+1][0]-gt[t][0]))
            gt[t].append((gt[t+1][1]-gt[t][1]))
        elif t-1 in gt:
            pass
        else:
            pass
    gt[1].append(0.)
    gt[1].append(0.)
    alex.update({g.id: gt})
    eval[g.id].update({"quota": float(len(auto))/len(gt)})
    dist = [np.linalg.norm(np.asarray(auto[t][:2])-np.asarray(gt[t][:2])) for t in gt if t in auto.keys()]
    eval[g.id].update({"mean_dist": np.mean(dist)})
    eval[g.id].update({"std_dist": np.std(dist)})
    treshold = 10
    while treshold > 0:
        auto_part = np.sum([a in auto for a in gt.keys() if np.linalg.norm(gt[a][2:]) > treshold])
        gt_part = np.sum([np.linalg.norm(a[2:]) > treshold for a in gt.values()])
        # print(auto_part, gt_part, float(auto_part)/gt_part, eval[g.id]["quota"])
        mq = float(auto_part)/gt_part
        if mq > eval[g.id]["quota"]:
            eval[g.id].update({"movement_quota": mq})
            print(treshold)
            break
        treshold -= 1
    else:
        eval[g.id].update({"movement_quota": eval[g.id]["quota"]})


with open("eval.txt","a") as myfile:
    myfile.write("\n")
    myfile.write("\n")
    myfile.write("\n")
    for k in eval:
        myfile.write("\n")
        myfile.write("Track %s"%k)
        myfile.write("\n")
        for a in eval[k]:
            myfile.write("%s: %s"%(a,eval[k][a]))
            myfile.write("\n")
    myfile.write("\n")
    for a in eval[1].keys():
        myfile.write("Mean %s: %s pm %s"%(a,np.mean([eval[k][a] for k in eval]),np.std([eval[k][a] for k in eval])))
        myfile.write("\n")
