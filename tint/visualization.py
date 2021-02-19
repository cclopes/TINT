"""
tint.visualization
==================

Visualization tools for tracks objects.

"""
import warnings
warnings.filterwarnings('ignore')

import gc
import os
import pandas as pd
import numpy as np
import shutil
import tempfile
import matplotlib as mpl
from IPython.display import display, Image
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

import pyart

from .grid_utils import get_grid_alt


class Tracer(object):
    colors = ['m', 'r', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']
    colors.reverse()

    def __init__(self, tobj, persist):
        self.tobj = tobj
        self.persist = persist
        self.color_stack = self.colors * 10
        self.cell_color = pd.Series()
        self.history = None
        self.current = None

    def update(self, nframe):
        self.history = self.tobj.tracks.loc[:nframe]
        self.current = self.tobj.tracks.loc[nframe]
        if not self.persist:
            dead_cells = [key for key in self.cell_color.keys()
                          if key
                          not in self.current.index.get_level_values('uid')]
            self.color_stack.extend(self.cell_color[dead_cells])
            self.cell_color.drop(dead_cells, inplace=True)

    def _check_uid(self, uid):
        if uid not in self.cell_color.keys():
            try:
                self.cell_color[uid] = self.color_stack.pop()
            except IndexError:
                self.color_stack += self.colors * 5
                self.cell_color[uid] = self.color_stack.pop()

    def plot(self, ax):
        for uid, group in self.history.groupby(level='uid'):
            self._check_uid(uid)
            tracer = group[['lon', 'lat']]
            if self.persist or (uid in self.current.index):
                ax.plot(tracer.lon, tracer.lat, self.cell_color[uid])

def full_domain(tobj, grids, tmp_dir, vmin=-8, vmax=64,
                cmap=None, alt=None, isolated_only=False,
                tracers=False, persist=False,
                projection=None, **kwargs):
    grid_size = tobj.grid_size
    if cmap is None:
        cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    if alt is None:
        alt = tobj.params['GS_ALT']
    if projection is None:
        projection=ccrs.PlateCarree()
    if tracers:
        tracer = Tracer(tobj, persist)

    radar_lon = tobj.radar_info['radar_lon']
    radar_lat = tobj.radar_info['radar_lat']
    lon = np.arange(round(radar_lon-5,2),round(radar_lon+5,2), 1)
    lat = np.arange(round(radar_lat-5,2),round(radar_lat+5,2), 1)

    nframes = tobj.tracks.index.levels[0].max() + 1
    print('Animating', nframes, 'frames')

    for nframe, grid in enumerate(grids):
        fig_grid = plt.figure(figsize=(10, 8))
        print('Frame:', nframe)
        display = pyart.graph.GridMapDisplay(grid)
        ax = fig_grid.add_subplot(111, projection=projection)
        transform = projection._as_mpl_transform(ax)
        display.plot_crosshairs(lon=radar_lon, lat=radar_lat)
        display.plot_grid(tobj.field, level=get_grid_alt(grid_size, alt),
                          vmin=vmin, vmax=vmax, mask_outside=False,
                          cmap=cmap, transform=projection, ax=ax, **kwargs)

        if nframe in tobj.tracks.index.levels[0]:
            frame_tracks = tobj.tracks.loc[nframe]

            if tracers:
                tracer.update(nframe)
                tracer.plot(ax)

            for ind, uid in enumerate(frame_tracks.index):
                if isolated_only and not frame_tracks['isolated'].iloc[ind]:
                    continue
                x = frame_tracks['lon'].iloc[ind]
                y = frame_tracks['lat'].iloc[ind]
                ax.text(x, y, uid, transform=projection, fontsize=20)


        plt.savefig(tmp_dir + '/frame_' + str(nframe).zfill(3) + '.png',
                    bbox_inches = 'tight', dpi=300)
        plt.close()
        del grid, display, ax
        gc.collect()


def lagrangian_view(tobj, grids, tmp_dir, uid=None, vmin=-8, vmax=64,
                    cmap=None, alt=None, box_rad=.1, projection=None):

    if uid is None:
        print("Please specify 'uid' keyword argument.")
        return
    # stepsize = 0.05
    title_font = 18
    axes_font = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16

    field = tobj.field
    grid_size = tobj.grid_size

    if cmap is None:
        cmap = pyart.graph.cm_colorblind.HomeyerRainbow
    if alt is None:
        alt = tobj.params['GS_ALT']
    if projection is None:
        projection = ccrs.PlateCarree()
        
    cell = tobj.tracks.xs(uid, level='uid')

    nframes = len(cell)
    print('Animating', nframes, 'frames')
    cell_frame = 0

    for nframe, grid in enumerate(grids):
        if nframe not in cell.index:
            continue

        print('Frame:', cell_frame)

        cell_frame += 1

        row = cell.loc[nframe]
        display = pyart.graph.GridMapDisplay(grid)

        # Box Size
        tx = np.int(np.round(row['grid_x']))
        ty = np.int(np.round(row['grid_y']))
        tx_met = grid.x['data'][tx]
        ty_met = grid.y['data'][ty]
        lat = row['lat']
        lon = row['lon']
        # print('lat = ' + str(lat))
        # print('lon = ' + str(lon))
        if np.sqrt(row['area'])/100. > box_rad:
            box_rad_met = np.sqrt(row['area'])/100.
        else:
            box_rad_met = box_rad
        box = np.array([-1*box_rad_met, box_rad_met])
        stepsize = box_rad_met*2/5

        lvxlim = (lon) + box
        lvylim = (lat) + box
        # print('lvxlim = ' + str(lvxlim[0]) + ' - ' + str(lvxlim[1]))
        # print('lvylim = ' + str(lvylim[0]) + ' - ' + str(lvylim[1]))
        xlim = (tx_met + np.array([-box_rad_met*1e5, box_rad_met*1e5]))/1000
        ylim = (ty_met + np.array([-box_rad_met*1e5, box_rad_met*1e5]))/1000
        # print('xlim = ' + str(xlim[0]) + ' - ' + str(xlim[1]))
        # print('ylim = ' + str(ylim[0]) + ' - ' + str(ylim[1]))
        lvxticks = np.arange(lvxlim[0], lvxlim[1], stepsize)
        lvyticks = np.arange(lvylim[0], lvylim[1], stepsize)
        llxticks = np.arange(xlim[0], xlim[1], stepsize*1e2)
        llyticks = np.arange(ylim[0], ylim[1], stepsize*1e2)
        # print(len(lvyticks), len(llyticks))

        fig = plt.figure(figsize=(15, 19))

        fig.suptitle('Cell ' + uid + ' Scan ' + str(nframe) + \
            '\n' + str(pyart.util.datetime_from_grid(grid)), fontsize=22)
        # plt.axis('off')

        gs = fig.add_gridspec(3, 3, wspace=0.3, hspace=0.25,
                              width_ratios=[1, 2, 0.005], 
                              height_ratios=[1, 2, 1])

        # Lagrangian View
        ax1 = fig.add_subplot(gs[1, 1:], projection=projection)

        display.plot_grid(field, level=get_grid_alt(grid_size, alt),
                          vmin=vmin, vmax=vmax, embelish=False,
                          mask_outside=False, cmap=cmap, colorbar_flag=False, 
                          ax=ax1, fig=fig, projection=projection)

        display.plot_crosshairs(lon=lon, lat=lat, linestyle='--', 
                                color='k', linewidth=3, ax=ax1)
        
        gl = ax1.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            xlocs=np.round(lvxticks, 2), ylocs=np.round(lvyticks, 2)
        )
        # gl.top_labels = gl.right_labels = False

        ax1.set_xlim(lvxlim[0], lvxlim[1])
        ax1.set_ylim(lvylim[0], lvylim[1])

        ax1.set_xticks(lvxticks)
        ax1.set_yticks(lvyticks)
        ax1.set_xticklabels(np.round(lvxticks, 2))
        ax1.set_yticklabels(np.round(lvyticks, 2))

        ax1.set_title('Top-Down View at ' + 
                      str(alt/1000) +
                      ' km', fontsize=title_font)
        ax1.set_xlabel('Longitude [° E]',
                       fontsize=axes_font)
        ax1.set_ylabel('Latitude [° N]',
                       fontsize=axes_font)
        cb = fig.colorbar(display.mappables[-1], ax=ax1, fraction=0.05, 
                          orientation='vertical')
        cb.set_label(
            grid.fields[field]['standard_name'].replace('_', ' ').title() + 
            ' (' + grid.fields[field]['units'] + ')', fontsize=axes_font)

        # Latitude Cross Section
        ax2 = fig.add_subplot(gs[0, 1])
        display.plot_latitude_slice(field, lon=lon, lat=lat,
                                    title_flag=False,
                                    colorbar_flag=False, edges=False,
                                    vmin=vmin, vmax=vmax, mask_outside=False,
                                    cmap=cmap, axislabels_flag=False,
                                    ax=ax2)
        ax2.grid()
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_xticks(llxticks)
        ax2.set_xticklabels(np.round(llxticks, 0).astype(int))

        ax2.set_title('Latitudinal Cross Section', fontsize=title_font)
        ax2.set_xlabel('East West Distance from Origin (km)', 
                       fontsize=axes_font)
        ax2.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
        # ax2.set_aspect(aspect=0.3)

        # Longitude Cross Section
        ax3 = fig.add_subplot(gs[1, 0])
        trans = mpl.transforms.Affine2D().rotate_deg(90) + ax3.transData

        display.plot_longitude_slice(field, lon=lon, lat=lat,
                                     title_flag=False, transform=trans,
                                     colorbar_flag=False, edges=False,
                                     vmin=vmin, vmax=vmax, mask_outside=False,
                                     cmap=cmap,     
                                     ax=ax3)
        ax3.grid()
        ax3.set_ylim(ylim[0], ylim[1])
        ax3.set_yticks(llyticks)
        ax3.set_yticklabels(np.round(llyticks, 0).astype(int))
        ax3.set_xticklabels([20, 15, 10, 5])

        ax3.set_title('Longitudinal Cross Section', fontsize=title_font)
        ax3.set_ylabel('North South Distance from Origin (km)', 
                       fontsize=axes_font)
        ax3.set_xlabel('Distance Above Origin (km)', fontsize=axes_font)
        # ax3.set_aspect(aspect=1.7)

        # Time Series Statistic
        max_field = cell['max']
        plttime = cell['time']

        # Plot
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(plttime, max_field, color='b', linewidth=3)
        ax4.axvline(x=plttime[nframe], linewidth=4, color='r')
        ax4.grid()
        ax4.set_title('Time Series', fontsize=title_font)
        ax4.set_xlabel('Time (UTC) \n Lagrangian Viewer Time',
                       fontsize=axes_font)
        ax4.set_ylabel('Maximum ' + field, fontsize=axes_font)

        plt.subplots_adjust(top=0.92)

        # plot and save figure
        fig.savefig(
            tmp_dir + '/frame_' + str(nframe).zfill(3) + '.png',
            bbox_inches='tight'
        )
        plt.close()
        del grid, display
        gc.collect()


def make_mp4_from_frames(tmp_dir, dest_dir, basename, fps):
    os.chdir(tmp_dir)
    os.system(" ffmpeg -framerate " + str(fps)
              + " -pattern_type glob -i '*.png'"
              + " -movflags faststart -pix_fmt yuv420p -vf"
              + " 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -y "
              + basename + '.mp4')
    try:
        shutil.move(basename + '.mp4', dest_dir)
    except FileNotFoundError:
        print('Make sure ffmpeg is installed properly.')


def animate(tobj, grids, outfile_name, style='full', fps=1, keep_frames=False,
            overwrite=False, **kwargs):
    """
    Creates gif animation of tracked cells.

    Parameters
    ----------
    tobj : Cell_tracks
        The Cell_tracks object to be visualized.
    grids : iterable
        An iterable containing all of the grids used to generate tobj.
    outfile_name : str
        The name of the output file to be produced.
    alt : float
        The altitude to be plotted in meters.
    vmin, vmax : float
        Limit values for the colormap.
    arrows : bool
        If True, draws arrow showing corrected shift for each object. Only used
        in 'full' style.
    isolation : bool
        If True, only annotates uids for isolated objects. Only used in 'full'
        style.
    uid : str
        The uid of the object to be viewed from a lagrangian persepective. Only
        used when style is 'lagrangian'.
    fps : int
        Frames per second for output gif.
    overwrite : bool
        If true, will overwrite existing mp4 if one already exists.
        False, won't overwrite if file already exists.

    """

    styles = {'full': full_domain,
              'lagrangian': lagrangian_view}
    anim_func = styles[style]

    dest_dir = os.path.dirname(outfile_name)
    basename = os.path.basename(outfile_name)
    if len(dest_dir) == 0:
        dest_dir = os.getcwd()

    if os.path.exists(basename + '.mp4') and overwrite is False:
        print('Filename already exists.')
        return

    tmp_dir = tempfile.mkdtemp()

    try:
        anim_func(tobj, grids, tmp_dir, **kwargs)
        if len(os.listdir(tmp_dir)) == 0:
            print('Grid generator is empty.')
            return
        make_mp4_from_frames(tmp_dir, dest_dir, basename, fps)
        if keep_frames:
            frame_dir = os.path.join(dest_dir, basename + '_frames')
            shutil.copytree(tmp_dir, frame_dir)
            os.chdir(dest_dir)
    finally:
        shutil.rmtree(tmp_dir)


def embed_mp4_as_gif(filename):
    """ Makes a temporary gif version of an mp4 using ffmpeg for embedding in
    IPython. Intended for use in Jupyter notebooks. """
    if not os.path.exists(filename):
        print('file does not exist.')
        return

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    newfile = tempfile.NamedTemporaryFile()
    newname = newfile.name + '.gif'
    if len(dirname) != 0:
        os.chdir(dirname)

    os.system('ffmpeg -i ' + basename + ' ' + newname)

    try:
        with open(newname, 'rb') as f:
            display(Image(f.read(), format='png'))
    finally:
        os.remove(newname)
