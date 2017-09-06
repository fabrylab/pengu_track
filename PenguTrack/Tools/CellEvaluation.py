from PenguTrack.DataFileExtended import DataFileExtended
import numpy as np

db = DataFileExtended("/home/alex/Masterarbeit/Data/Cells/DataBases/PT_Cell_T850_A75_inf_3d_backup.cdb")
tracks = db.getTracks(type="PT_Track_Marker")


# db = DataFileExtended("/home/alex/Desktop/PT_Cell_GT_Track.cdb")
# tracks = db.getTracks(type="GroundTruth")
V=[]
vals = {}
for track in tracks:
    if track.markers.count()<4:
        continue
    x,y,z,t = np.array([[m.measurement[0].x*0.646, m.measurement[0].y*0.645, m.measurement[0].z*0.645, m.image.timestamp] for m in track.markers]).T
    # x,y,z,t = np.array([[m.x*0.646, m.y*0.645, 0, m.image.timestamp] for m in track.markers]).T
    n=10.
    # z = np.convolve(z, np.ones(n)/n, mode="same") if len(z)>n else z
    t = (np.array([tt.total_seconds() for tt in t[1:]-t[:-1]], dtype=float))
    vx = ((x[1:]-x[:-1])/t).astype(float)
    vy = ((y[1:]-y[:-1])/t).astype(float)
    vz = ((z[1:]-z[:-1])/t).astype(float)
    # vz = np.convolve(vz, np.ones(10)/10., mode="same") if len(vz)>10 else vz

    # v = (vx**2+vy**2+vz**2)**0.5
    v = (vx**2+vy**2)**0.5
    print(np.nanmean(v), np.mean(np.abs(vz)), np.mean(vx), np.mean(vy))
    w1 = np.array([vx[:-1],vy[:-1],vz[:-1]]).T
    e_1 = (w1.T/np.linalg.norm(w1, axis=1)).T
    w2 = np.array([vx[1:],vy[1:],vz[1:]]).T
    e2_ = w2 - (np.diag(np.tensordot(e_1,w2, axes=[[1],[1]]))*e_1.T).T
    e_2 = (e2_.T/np.linalg.norm(e2_, axis=1)).T
    # print(e_2.shape)
    # print(np.diag(np.tensordot(e_1,w2, axes=[[1],[1]])).shape)
    w3 = np.array([vy[:-1]*vz[1:]-vz[:-1]*vy[1:],
                   vz[:-1]*vx[1:]-vx[:-1]*vz[1:],
                   vx[:-1]*vy[1:]-vy[:-1]*vx[1:]]).T
    e3_ = w3 - (np.diag(np.tensordot(e_1,w3, axes=[[1],[1]]))*e_1.T).T - (np.diag(np.tensordot(e_2, w3, axes=[[1],[1]]))*e_2.T).T
    e3 = (e3_.T/np.linalg.norm(e3_, axis=1)).T

    ang = np.arccos(np.diag(np.tensordot(e_1[1:], e_1[:-1], axes=[[1],[1]])))*np.sign(e_1[1:,1]-e_1[:-1,1])
    ang2 = np.arccos(np.diag(np.tensordot(e_2[1:], e_2[:-1], axes=[[1],[1]])))*np.sign(e_2[1:,1]-e_2[:-1,1])
    # ang = ang1*np.cos(ang2)



    # ang = np.arctan2((vx[1:]*vx[:-1]+vy[1:]*vy[:-1]),(vy[1:]*vx[:-1]-vx[1:]*vy[:-1]))
    ang = np.arccos((vx[1:]*vx[:-1]+vy[1:]*vy[:-1]+vz[1:]*vz[:-1])/(v[1:]*v[:-1]))
    # ang = np.arccos(np.nanmean(vx[1:]/v[1:])*np.nanmean(vx[:-1]/v[:-1])+
    #                 np.nanmean(vy[1:]/v[1:])*np.nanmean(vy[:-1]/v[:-1])+
    #                 np.nanmean(vz[1:]/v[1:])*np.nanmean(vz[:-1]/v[:-1]))
    # motile = np.linalg.norm([x[-1]-x[0],y[-1]-y[0]])>12
    motile = np.array(np.amax(((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)**0.5)>12) if len(x)>3 else np.array(np.any(((vx**2+vy**2)**0.5)>12))
    # motile = np.linalg.norm([x[-1]-x[0],y[-1]-y[0]])>60

    motillity = np.sum(v>0.32).astype(float)/len(v)
    print(np.nanmean(ang))
    if not (np.isnan(np.nanmean(ang)) or np.isnan(np.nanmean(v))):
        vals.update({track.id:[np.nanmean(v), np.nanstd(v), 180/np.pi*np.nanmean(ang), np.nanstd(180/np.pi*ang), motile.astype(bool), motillity]})
    # if len(vx)>20:
        V.append(np.linalg.norm([vx,vy],axis=0))
        # V.append(np.linalg.norm(np.mean([[vx[i:i+20],vy[i:i+20]] for i in range(len(vx)-20)],axis=2),axis=0))

v, v_std, ang, ang_std, motile, motillity = np.array(vals.values()).T
V=np.hstack(V)
vv = np.array([np.mean(val) for val in V])
import matplotlib.pyplot as plt
import my_plot
import seaborn as sn

color = [("red" if m else "green" )for m in motile]
motile = np.array(motile, dtype=bool)
fig, ax = plt.subplots()
ax.set_title("Motility of T-Cells")
ax.set_xlabel(r"Average Speed of Cell in $\frac{\mathrm{\mu m}}{\mathrm{min}}$")
ax.set_ylabel(r"Average Turning Angle in Degrees")
# ax.set_xlim([2,100])
# ax.set_ylim([40,140])
ax.semilogx()
# ax.scatter(v*60,ang,label="System Tracks")
ax.scatter(60*v[motile],ang[motile],label="System Tracks (motile)")
ax.scatter(60*v[~motile],ang[~motile],label="System Tracks (immotile)")
ax.legend(loc="best")
my_plot.despine(ax)
my_plot.setAxisSizeMM(fig, ax, width=147, height=90)
plt.savefig("/home/alex/Desktop/Cell_Classification.png")
plt.savefig("/home/alex/Desktop/Cell_Classification.pdf")
# plt.semilogx()

fig, ax = plt.subplots()
ax.set_title("Velocity of T-Cells")
ax.set_ylabel("Absolute Frequency")
ax.set_xlabel(r"Velocity in $\frac{\mathrm{\mu m}}{\mathrm{min}}$")
hist, bins = np.histogram(np.log(V*60), bins=200, range=np.log([1e-3,1e2]))
ax.hist(V*60, bins=np.exp(bins))
ax.semilogx()


fig, ax = plt.subplots()
ax.set_title("Velocity of T-Cells")
ax.set_ylabel("Absolute Frequency")
ax.set_xlabel(r"Averaged Velocity in $\frac{\mathrm{\mu m}}{\mathrm{min}}$")
hist, bins = np.histogram(np.log(vv*60), bins=50, range=np.log([1e-3,1e2]))
ax.hist(vv*60, bins=np.exp(bins))
ax.semilogx()
#
#
