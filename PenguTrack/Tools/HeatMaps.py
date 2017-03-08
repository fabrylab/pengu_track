import clickpoints
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import peewee
from scipy.interpolate import griddata
from skimage.color import gray2rgb

db = clickpoints.DataFile("./241_tracked.cdb")

class Measurement(db.base_model):
    # full definition here - no need to use migrate
    marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name="measurement", on_delete='CASCADE') # reference to frame and track via marker!
    log = peewee.FloatField(default=0)
    x = peewee.FloatField()
    y = peewee.FloatField()

if "measurement" not in db.db.get_tables():
    db.db.connect()
    Measurement.create_table()#  important to respect unique constraint

db.table_measurement = Measurement   # for consistency


def setMeasurement(marker=None, log=None, x=None, y=None):
    assert not (marker is None), "Measurement must refer to a marker."
    try:
        item = db.table_measurement.get(marker=marker)
    except peewee.DoesNotExist:
        item = db.table_measurement()

    dictionary = dict(marker=marker, x=x, y=y)
    for key in dictionary:
        if dictionary[key] is not None:
            setattr(item, key, dictionary[key])
    item.save()
    return item

def getMeasurement(marker=None):
    assert not (marker is None), "Measurement must refer to a marker."
    item = db.table_measurement.get(marker=marker)
    return item

db.setMeasurement = setMeasurement
db.getMeasurement = getMeasurement


GT_Type = db.getMarkerType(name="GT")
GT_Tracks = db.getTracks(type=GT_Type)
GT_Data = {}
for t in GT_Tracks:
    GT_Data.update({t.id: {}})
    for m in db.getMarkers(type=GT_Type, track=t):
        GT_Data[t.id].update({m.image_id: [m.x, m.y]})

Auto_Type = db.getMarkerType(name="PT_Track_Marker")
Auto_Tracks = db.getTracks(type=Auto_Type)
Auto_Data = {}
Auto_Measure = {}
for t in Auto_Tracks:
    Auto_Data.update({t.id: {}})
    Auto_Measure.update({t.id: {}})
    for m in db.getMarkers(type=Auto_Type, track=t):
        try:
            Auto_Data[t.id].update({m.image_id: [m.x, m.y, float(m.text.split("Prob")[1])]})
        except AttributeError:
            pass
        try:
            meas = db.getMeasurement(m)
        except:
            print("No measurement for marker%s"%m.id)
        Auto_Measure[t.id].update({m.image_id: [meas.x, meas.y]})
Auto_Data.pop(0)
Auto_Measure.pop(0)

all_probs = []
[all_probs.extend(a.values()) for a in Auto_Data.values()]
all_probs = np.asarray(all_probs).T[2]
mean_probs = [np.mean(np.array(a.values(), ndmin=2).T[2], axis=0) for a in Auto_Data.values() if len(a.values()) > 3]

heaters_x = []
heaters_x0 = []
heaters_v = []
for a in Auto_Measure:
    if len(Auto_Measure[a])>3:
        heaters_x.extend(Auto_Measure[a].values())
        heaters_x0.extend(Auto_Measure[a].values()[:-1])
        heaters_v.extend(np.array(Auto_Measure[a].values())[1:, :]-np.array(Auto_Measure[a].values())[:-1, :])

heaters_x_img = []
for a in Auto_Data:
    if len(Auto_Data[a])>3:
        heaters_x_img.extend(Auto_Data[a].values())

heaters_x = np.asarray(heaters_x)
heaters_x0 = np.asarray(heaters_x0)
heaters_v = np.asarray(heaters_v)

heaters_x_img = np.asarray(heaters_x_img)

img = db.getImage(frame=0).data

x_max = np.amax(heaters_x.T[0])
x_min = np.amin(heaters_x.T[0])
y_max = np.amax(heaters_x.T[1])
y_min = np.amin(heaters_x.T[1])
img2 = np.zeros_like(img).T[0].T
hist, binsx, binsy = np.histogram2d(heaters_x.T[0], heaters_x.T[1], bins=[500,500])
# grid_x, grid_y = np.mgrid[x_min:x_max, y_min:y_max]
# xx, yy=np.meshgrid(binsx[:-1], binsy[:-1])
# grid_z0 = griddata(np.array([xx.flatten(), yy.flatten()]).T, hist.flatten(), (grid_x, grid_y), method='linear')
# # img2[y_min:y_max,x_min:x_max] = hist.astype(np.uint8).T
# plt.imshow(hist.T, extent=(x_max, 2*x_max, 0, y_max), cmap="coolwarm")
plt.imshow(hist, extent=(-250, 250, 0, 500), cmap="coolwarm", interpolation="bicubic")
plt.grid(False)
# plt.imshow(img, extent=(0, img.shape[1], 0, img.shape[0]))
# plt.show()

# plt.hist2d(heaters_x.T[0], heaters_x.T[1])

angle = np.arctan2(heaters_v.T[0], heaters_v.T[1])*(180/np.pi)

grid_x, grid_y = np.mgrid[0:500,250:750]#np.mgrid[int(x_min):int(x_max), int(y_min):int(y_max)]
grid_z0 = griddata(heaters_x0, angle, (grid_x, grid_y), method='linear')
grid_z0 -= np.nanmin(grid_z0)
grid_z0 *= ((np.amax(angle)-np.amin(angle))/(np.nanmax(grid_z0)-np.nanmin(grid_z0)))
grid_z0 += np.amin(angle)
plt.figure()
# heat_map = plt.imshow(grid_z0.T, extent=(y_min-250,y_max-250,x_min,x_max), cmap='coolwarm')
heat_map = plt.imshow(grid_z0, extent=(-250,250,0,500), cmap='coolwarm')#, interpolation="linear")
plt.colorbar(heat_map)

# plt.figure()
# plt.hist2d(heaters_x0.T[0], heaters_x0.T[1], bins=500, cmap="coolwarm")
plt.show()
# plt.contourf(heaters_x0.T[0], heaters_x0.T[1], np.array(angle, ndmin=2))
# plt.show()
# heat = plt.hist2d()