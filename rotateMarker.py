import clickpoints

db = clickpoints.DataFile("./Developement/Databases/click0.cdb")

Y, X = db.getImage(frame=0).getShape()

for marker in db.getMarkers():
    marker.x = X - marker.x
    marker.y = Y - marker.y
    # db.setMarker(id=marker.id, x=x, y=y)
    print(marker)
    marker.save()
