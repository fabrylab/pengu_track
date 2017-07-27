import clickpoints
import numpy as np

def copy_type(db1,db2,name):
    m_type=db1.getMarkerType(name=name)
    type = db2.setMarkerType(name=m_type.name, color=m_type.color, mode=m_type.mode, style=m_type.style, text=m_type.text,hidden=m_type.hidden)
    return type

def copy_tracks(db1,db2,track_ids):
    for id in track_ids:
        track = db1.getTrack(id=id)
        if not db2.getMarkerType(name=track.markers[0].type.name):
            copy_type(db1, db2, track.markers[0].type.name)
        type = db2.getMarkerType(name=track.markers[0].type.name)
        new_track = db2.setTrack(type, style=track.style, text=track.text, hidden=track.hidden)
        ids, filenames, x, y, text, layer = np.asarray([[m.image.id, m.image.filename, m.x, m.y, m.text, m.image.layer] for m in track.markers if db2.getImage(filename=m.image.filename) is not None]).T
        db2.setMarkers(image=[db2.getImage(filename=f) for f in filenames], x=x,y=y,type=type, track=new_track, text=text)

if __name__ == "__main__":
    db1 = clickpoints.DataFile("/home/user/Desktop/252/252Horizon.cdb")
    db2 = clickpoints.DataFile("/home/user/Desktop/PT_Test_LOG21.cdb")
    copy_type(db1, db2, "GT")
    copy_tracks(db1,db2,[1,2,3,4,5])