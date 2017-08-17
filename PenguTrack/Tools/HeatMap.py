if __name__ == '__main__':
    import my_plot
    import numpy as np
    import clickpoints
    import matplotlib.pyplot as plt
    from PenguTrack.DataFileExtended import DataFileExtended

    # Load Data from Databases
    db = DataFileExtended("/home/birdflight/Desktop/PT_Test_full_n3_r7_A20_filtered.cdb")
    db2 = DataFileExtended("/home/birdflight/Desktop/252_GT_Detections.cdb")
    LogType=db.getMarkerType(name="PT_Detection_Marker")
    ImgType=db.getMarkerType(name="PT_Track_Marker")
    # Positions_Log = np.asarray([[m.x, m.y, m.image.sort_index] for m in db.getMarkers(type=LogType) if not m.text.count("inf")])
    Positions_Img = np.asarray([[m.x, m.y, m.image.sort_index] for m in db.getMarkers(type=ImgType) if m.track.markers.count() >3])

    # Do Correction for Position Data
    from CameraTransform import CameraTransform
    CT = CameraTransform(14,[17,9],[4608,2592],observer_height=31.,angel_to_horizon=(np.pi/2.-0.24)*180/np.pi)
    orth_x, orth_y, orth_z = CT.transCamToWorld(Positions_Img.T[:2], Z=0.525)

    # Calculate Histogramms
    cx=cy=2
    times = np.asarray(sorted([i.timestamp for i in db.getImages()]))
    scale=1./(cx*cy)/((times[-1]-times[0]).seconds/3600.)
    hist, binx, biny = np.histogram2d(orth_x,orth_y, bins=[int(max(orth_x)-min(orth_x))/cx,int(max(orth_y)-min(orth_y))/cy], range=[[min(orth_x),max(orth_x)],[min(orth_y),max(orth_y)]])
    hist *= scale
    hist[hist==0.]=np.nan

    cx=cy=6
    hist_img, binx_img, biny_img = np.histogram2d(Positions_Img.T[0], Positions_Img.T[1], bins=[4608/cx, 2592/cy], range=[[0, 4608],[0,2592]])
    hist_img *= scale*cx*cy
    hist_img[hist_img==0.]=np.nan

    # beauty-corrections for background image
    from skimage.exposure import equalize_adapthist, adjust_gamma
    image = equalize_adapthist(db.getImage(0).data)
    image = adjust_gamma(image, gamma=.5)

    # Plotting
    fig, ax = plt.subplots()
    ax.set_title("Position Heatmap")
    ax.set_xlabel("X-Distance to camera in m")
    ax.set_ylabel("Y-Distance to camera in m")
    ax.imshow(CT.getTopViewOfImage(image, [-max(binx),-min(binx),min(biny),max(biny)]), extent=[-max(binx),-min(binx),min(biny),max(biny)])
    hist_plot = ax.imshow(hist.T[::-1, ::-1], extent=[-max(binx),-min(binx),min(biny),max(biny)], cmap="jet", alpha=0.4,vmin =1., vmax=max(2,np.nanmean(hist)+np.nanstd(hist)))
    cbar = fig.colorbar(hist_plot)
    cbar.ax.set_ylabel(r"Penguin Probability in $\frac{1}{\mathrm{m}^2\mathrm{h}}$")
    my_plot.setAxisSizeMM(fig, ax, 147)
    plt.savefig("/home/birdflight/Desktop/Heat_Map.png")
    plt.savefig("/home/birdflight/Desktop/Heat_Map.pdf")


    fig, ax = plt.subplots()
    ax.set_title("Position Heatmap")
    # ax.set_xlabel("X-Distance to camera in m")
    # ax.set_ylabel("Y-Distance to camera in m")
    ax.imshow(image[::-1])
    hist_plot = ax.imshow(hist_img.T, extent=[0, 4608, 0, 2592], cmap="jet", alpha=.4)#,vmin =1., vmax=max(2,np.nanmean(hist)+np.nanstd(hist)))
    # cbar = fig.colorbar(hist_plot)
    # cbar.ax.set_ylabel(r"Penguin Probability in $\frac{1}{\mathrm{m}^2\mathrm{h}}$")
    my_plot.setAxisSizeMM(fig, ax, 147)
    plt.savefig("/home/birdflight/Desktop/Heat_Map_Img.png")
    plt.savefig("/home/birdflight/Desktop/Heat_Map_Img.pdf")

    plt.show()
