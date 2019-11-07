#!/home/matthew/anaconda3/bin python3

import sys
import math
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import cm
import line as li
import plane
import utils_linear as ul
import utils_functions as uf
import normalize as nz
from scipy import special
from decimal import Decimal

class Field():
    """ CLASS Field()
    field object to contain a scalar field (vector field might be implemented here in the future
    too) and all the associated information. Storage assumes linear three-dimensional space. The
    contents are stored in an integer cartesian space and a transformation is stored for conversion
    to and from linear space.

    VARIABLES
    np.array((3), int):         f_size, an array giving the dimensions of the field
    np.array((i, j, k), float): f_field, an array (initialized to NoneType since this cannot be allocated
                                until size is known) giving the contents of the field
    np.array((3), float):       f_orig, array that stores the origin point
    np.array((3, 3), float):    f_t, matrix that stores the transformation (from field space to cartesian space
    int:                        nAtoms, the number of atoms in the cube file
    np.array(int):              a_AtomicNumber, list of atomic numbers in the order specified in the cube file
    np.array(int):              a_Charge, list of atomic charges in the order specified in the cube file
    np.array(float, float):     a_Pos, matrix of the atomic positions

    CONSTRUCTORS
    NONE
    """
    def __init__(self):
        self.f_size = np.zeros((3,), dtype=int)
        self.f_field = None
        self.f_orig = np.zeros((3,), dtype=float)
        self.f_t = np.zeros((3, 3,), dtype=float)
        self.nAtoms = 0
        self.nPoints = 0
        self.a_AtomicNumber = None
        self.a_Charge = None
        self.a_Pos = None

    def read_gc(self, fpath):
        """ FUNCTION read_gc(string: fpath)
        reads a cube input (assumes qe cube input) line by line

        INPUTS
        string/path:        relative path of the cube file that needs reading
        """
        with open(fpath) as f:
            fx = 0
            fy = 0
            fz = 0
            for idx, line in enumerate(f):
                #print(line)
                if idx < 2:
                    continue
                if idx == 2:
                    # line that contains number of atoms and position of origin
                    l = re.split('\s+', line)
                    #print(l)
                    #print(l[1])
                    self.nAtoms = int(l[1])
                    self.a_AtomicNumber = np.zeros((self.nAtoms), dtype=int)
                    self.a_Charge = np.zeros((self.nAtoms), dtype=float)
                    self.a_Pos = np.zeros((self.nAtoms, 3), dtype=float)
                    self.f_orig[0] = l[2]
                    self.f_orig[1] = l[3]
                    self.f_orig[2] = l[4]
                    continue
                if idx < 6:
                    # next three lines contain size and transform matrix
                    l = re.split('\s+', line)
                    #print(l)
                    self.f_size[idx-3] = l[1]
                    self.f_t[idx-3, 0] = l[2]
                    self.f_t[idx-3, 1] = l[3]
                    self.f_t[idx-3, 2] = l[4]
                    if idx == 5:
                        self.f_t = self.f_t.transpose()
                        self.nPoints = np.prod(self.f_size)
                        self.f_field = np.zeros((self.f_size), dtype=float)
                    continue
                if idx < (6 + self.nAtoms):
                    l = re.split('\s+', line)
                    self.a_AtomicNumber[idx-6] = l[1]
                    self.a_Charge[idx-6] = l[2]
                    self.a_Pos[idx-6, 0] = l[3]
                    self.a_Pos[idx-6, 1] = l[4]
                    self.a_Pos[idx-6, 2] = l[5]
                if idx > (5 + self.nAtoms):
                    l = re.split('\s+', line)[:-1]
                    for e in l:
                        if e is None or e == '':
                            continue
                        elif e == 'NaN':
                            self.f_field[fx, fy, fz] = 0.0
                        else:
                            self.f_field[fx, fy, fz] = e
                        fz += 1
                        if fz >= self.f_size[2]:
                            fz = 0
                            fy += 1
                            if fy >= self.f_size[1]:
                                fy = 0
                                fx += 1
                                if fx >= self.f_size[0]:
                                    break
                                
                    continue
        return

    def read_atoms(self, fpath):
        """ FUNCTION read_atoms(string: fpath)
        reads a cube input (assumes qe cube input) and returns a np.array of the atom positions
        """
        with open(fpath) as f:
            nAtoms = 0
            atoms = None
            for idx, line in enumerate(f):
                if idx == 2:
                    l = re.split('\s+', line)
                    nAtoms = int(l[1])
                    atoms = np.zeros((nAtoms, 3), dtype=float)
                if idx > 5 and idx < (6 + nAtoms):
                    l = re.split('\s+', line)
                    atoms[idx-6, 0] = l[3]
                    atoms[idx-6, 1] = l[4]
                    atoms[idx-6, 2] = l[5]
            return atoms

    def write_gc(self, fpath):
        """ FUNCTION write_gc(string: fpath)
        writes a cube file of the current stored information if possible

        INPUT:
        fpath, the path of the output file

        TODO:
        currently this function writes cube files that are a lot larger than they should be, check why this is the
        case and fix. 
        """
        if self.f_field is None:
            print("WARNING: field has not been created, nothing to write. write_gc will return without doing anything")
            return

        with open(fpath, 'w+') as f:
            f.write("CUBE FILE GENERATED BY FIELD.PY\n")
            f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
            f.write("%5i%12.6f%12.6f%12.6f\n" % (self.nAtoms, self.f_orig[0], self.f_orig[1], self.f_orig[2]))
            for idx in np.arange(3):
                f.write("%5i%12.6f%12.6f%12.6f\n" % (self.f_size[idx],
                                                     self.f_t[idx, 0], self.f_t[idx, 1], self.f_t[idx, 2]))
            for idx in np.arange(self.nAtoms):
                f.write("%5i%12.6f%12.6f%12.6f%12.6f\n" % (self.a_AtomicNumber[idx],
                                                           self.a_Charge[idx],
                                                           self.a_Pos[idx, 0], self.a_Pos[idx, 1], self.a_Pos[idx, 2]))
            i_field = 0
            t_field = self.f_size[0] * self.f_size[1] * self.f_size[2]
            while(i_field < t_field):
                l_str = 6
                if (t_field - i_field) < 6:
                    l_str = t_field - i_field
                t_str = ""
                for idx in np.arange(l_str):
                    x_str = ("%12.5E" % (self.f_field[self.__n2f(i_field)]))
                    t_str += self.__fixexp(x_str)
                    t_str += " "
                    i_field += 1
                t_str = t_str[:-1]
                t_str += "\n"
                f.write(t_str)

    def __n2f(self, n):
        x = n // (self.f_size[1] * self.f_size[2])
        rx = n % (self.f_size[1] * self.f_size[2])
        y = rx // self.f_size[2]
        z = rx % self.f_size[2]
        return (x, y, z)

    def __fixexp(self, n):
        # shift the decimal point
        n = re.sub(r"(\d)\.", r".\1", n)
        # add 0 for positive numbers
        n = re.sub(r" \.", r"0.", n)
        # increase exponent by 1
        exp = re.search(r"E([+-]\d+)", n)
        newexp = "E{:+03}".format(int(exp.group(1))+1)
        n = re.sub(r"E([+-]\d+)", newexp, n)
        return n

    def init_size(self, x, y, z):
        assert(isinstance(x, int))
        assert(isinstance(y, int))
        assert(isinstance(z, int))
        self.f_size[0] = x
        self.f_size[1] = y
        self.f_size[2] = z
        self.f_field = np.zeros((x, y, z), dtype=float)
        self.f_t = np.eye(3)

    def init_shape(self, shape=0, arg1=1, arg2=0.2):
        """ FUNCTION init_shape(int(optional) shape, float(optional) arg1, float(optional) arg2)
        builds a shape as a field with 1 inside and 0 outside

        INPUT:
        int:            shape to be created (should maybe be an enum)
                        1 = SPHERE
        float:          argX where X is an integer, the parameters for the shape

        TODO: change to kwargs
        """
        if shape == 0:
            # SPHERE
            radius = arg1
            smoothing = arg2
            # transform sphere if f_t set
            if np.any(self.f_t):
                # sphere should be centered properly, get the center of the grid
                # space in field space and transform this point. Set this point
                # as a temporary reference origin point used to define the sphere
                # equation.
                c_idx = np.rint(self.f_size / 2)
                origin = np.matmul(self.f_t, c_idx)
                c1 = 1
                c2 = radius
                c3 = smoothing
                if c2 == 0:
                    nu = 1
                elif c3 == 0:
                    nu = 4 * np.pi * (c1 * c2) ** 3 / 3
                else:
                    f0 = (c3 / c2) ** 2
                    f1 = (1.0 + special.erf(c2 / c3)) / 2.0
                    f2 = np.exp(-(c2 / c3) ** 2) / (2.0 * np.sqrt(np.pi))      
                    correction = (1.0 + 1.5 * f0) * f1 + (1.0 + f0) * (c3 / c2) * f2
                    nu = 4 * np.pi * correction * (c1 * c2) ** 3 / 3
                for idx in np.ndindex(self.f_field.shape):
                    # transform idx to cartesian space
                    i = np.matmul(self.f_t, idx)
                    # get distance to origin
                    d2 = uf.vect_distance(i, origin)
                    # smoothing erfc from 0 outside to 1 inside
                    self.f_field[idx] = uf.sigmoid(uf.SigType.erfc, np.sqrt(d2),
                                                 c1, c2, c3) / nu
                    #if d2 < radius ** 2:
                    #    self.f_field[idx] = 1\
                self.norm_field()
        elif shape == 1:
            # QUAD SPHERE
            radius = arg1
            if arg2 == 0:
                arg2 = 1
            smoothing = 1 / arg2
            if np.any(self.f_t):
                c_idx = np.rint(self.f_size / 2)
                origin = np.matmul(self.f_t, c_idx)
                if radius == 0:
                    radius = 1
                for idx in np.ndindex(self.f_field.shape):
                    i = np.matmul(self.f_t, idx)
                    d2 = uf.vect_distance(i, origin)
                    self.f_field[idx] = 1 / ((d2 / (smoothing ** 2)) + radius)
        elif shape == 2:
            # PLANE increasing in x-axis to test grad function
            for idx in np.ndindex(self.f_field.shape):
                self.f_field[idx] = idx[0]**2 + idx[1]**2 + idx[2]**2

    def write_npz(self, fname):
        """ FUNCTION write_npz(string: fname)
        Writes to a less space efficient file that can be loaded in quickly
        """
        f = open(fname, 'wb')
        np.savez(f, self.f_size, self.f_orig, self.f_t, self.f_field)

    def read_npz(self, fname):
        """ FUNCTION read_npz(string: fname)
        Reads in a npz file
        """
        f = open(fname, 'rb')
        npzfile = np.load(f)
        self.f_size = npzfile['arr_0']
        self.f_orig = npzfile['arr_1']
        self.f_t = npzfile['arr_2']
        self.f_field = np.zeros((self.f_size), dtype=float)
        self.f_field = npzfile['arr_3']

    def get_totfield(self):
        tot = 0.
        for val in np.nditer(self.f_field):
            tot += val
        return tot

    def norm_field(self):
        s = np.sum(self.f_field)
        for idx in np.ndindex(self.f_field.shape):
            self.f_field[idx] /= s

    def trans_point_to_space(self, point):
        """ FUNCTION trans_point_to_space
        transforms a point (or a vector) from the cartesian basis to the modified basis

        INPUT:
        np.array((3), float):       point, a 3-vec that describes the point in the cartesian basis

        OUTPUT:
        np.array((3), float):       a 3-vec that describes the point in the modified basis
        """
        return np.matmul(np.linalg.inv(self.f_t), point) + self.f_orig

    def trans_point_from_space(self, point):
        """ FUNCTION trans_point_from_space
        transforms a point (or a vector) from the modified basis to the cartesian basis

        INPUT:
        np.array((3), float):       point, a 3-vec that describes the point in the modified basis

        OUTPUT:
        np.array((3), float):       a 3-vec that describes the point in the cartesian basis
        """
        return np.matmul(self.f_t, point) - self.f_orig

    def has_point(self, point):
        """ FUNCTION has_point
        determines whether a given point is inside the vector space

        INPUT:
        np.array((3), float):       point, a 3-vec that describes the point in the modified basis

        OUTPUT:
        bool:                       True if the point lies in the vector space
                                    False if the point lies outside the space
        """
        # assumes the point is in the same space as the field
        inside = True
        pos = np.round(point)
        if pos[0] < 0 or pos[0] > float(self.f_size[0]):
            inside = False
        if pos[1] < 0 or pos[1] > float(self.f_size[1]):
            inside = False
        if pos[2] < 0 or pos[2] > float(self.f_size[2]):
            inside = False
        return inside

    def _get_position(self, basis, val, line):
        """ PRIVATE FUNCTION _get_position
        returns the intersection between the given line and a plane defined by the basis set of the defined vector
        space

        INPUT:
        integer:        basis, a value between 0 and 2 that specifies the direction normal to the desired plane
        float:          val, the point along the chosen basis that defines the plane
        line:           line, the line to intersect with the plane
        """
        return line.get_position(basis, val)

    def has_line(self, line):
        """ FUNCTION has_line
        determines if a line is in the bounds of the field space and if yes, returns a list of the two points
        assumes the line is in the same space as the field

        INPUT:
        line:           line, the line to check

        OUTPUT:
        bool:           valid,  True if the line is inside
                                False if the line is outside
        list(np.array((3), float)):     points, empty if the line is outside, otherwise contains two 3-vecs describing
                                        the points where the line intersects with the boundaries of the field
        """
        valid = False
        points = []
        for i in np.arange(3):
            if not line.dir[i] == 0:
                t = self._get_position(i, 0, line)
                if self.has_point(t):
                    valid = True
                    points.append(t)
                u = self._get_position(i, self.f_size[i], line)
                if self.has_point(u):
                    valid = True
                    points.append(u)
        return valid, points

    def remove_negative(self):
        for idx, val in np.ndenumerate(self.f_field):
            if val < 0.0:
                self.f_field[idx] = 0.0

    def truncate_range(self, fmin, fmax):
        for idx, val in np.ndenumerate(self.f_field):
            if val < fmin:
                self.f_field[idx] = fmin
            elif val > fmax:
                self.f_field[idx] = fmax

    def rescale_range(self, axis, fmin, fmax):
        """ linearly rescales each plane to 0, 1
        """
        for idx in np.arange(self.f_size[axis]):
            if axis == 0:
                for fdx, fval in np.ndenumerate(self.f_field[idx,:,:]):
                    tdx = (idx, fdx[0], fdx[1])
                    self.f_field[tdx] = (fval - fmin) / (fmax - fmin)
            if axis == 1:
                for fdx, fval in np.ndenumerate(self.f_field[:,idx,:]):
                    tdx = (fdx[0], idx, fdx[1])
                    self.f_field[tdx] = (fval - fmin) / (fmax - fmin)
            if axis == 2:
                for fdx, fval in np.ndenumerate(self.f_field[:,:,idx]):
                    tdx = (fdx[0], fdx[1], idx)
                    self.f_field[tdx] = (fval - fmin) / (fmax - fmin)

    def rescale(self, axis):
        """ linearly rescales each plane such that the max value is 1
        """
        for idx in np.arange(self.f_size[axis]):
            if axis == 0:
                scale = np.amax(self.f_field[idx,:,:])
                if scale == 0:
                    continue
                for fdx, fval in np.ndenumerate(self.f_field[idx,:,:]):
                    tdx = (idx, fdx[0], fdx[1])
                    self.f_field[tdx] = fval / scale
            if axis == 1:
                scale = np.amax(self.f_field[:,idx,:])
                if scale == 0:
                    continue
                for fdx, fval in np.ndenumerate(self.f_field[:,idx,:]):
                    tdx = (fdx[0], idx, fdx[1])
                    self.f_field[tdx] = fval / scale
            if axis == 2:
                scale = np.amax(self.f_field[:,:,idx])
                if scale == 0:
                    continue
                for fdx, fval in np.ndenumerate(self.f_field[:,:,idx]):
                    tdx = (fdx[0], fdx[1], idx)
                    self.f_field[tdx] = fval / scale

    def interpolate(self, point):
        """ FUNCTION interpolate
        use trilinear interpolation to get an approximate value of the field at an exact point

        INPUT:
        np.array((3), float):       point

        OUTPUT:
        float:                      vxyz, the field at that point
        """
        fpoint = np.floor(point)
        x = int(fpoint[0])
        y = int(fpoint[1])
        z = int(fpoint[2])
        x1 = x + 1
        y1 = y + 1
        z1 = z + 1
        if x1 >= self.f_size[0]:
            x1 = x
        if y1 >= self.f_size[1]:
            y1 = y
        if z1 >= self.f_size[2]:
            z1 = z
        v000 = self.f_field[x, y, z]
        v001 = self.f_field[x, y, z1]
        v010 = self.f_field[x, y1, z]
        v011 = self.f_field[x, y1, z1]
        v100 = self.f_field[x1, y, z]
        v101 = self.f_field[x1, y, z1]
        v110 = self.f_field[x1, y1, z]
        v111 = self.f_field[x1, y1, z1]
        npoint = point - fpoint

        #reduce x
        vx00 = v000 + npoint[0] * (v100 - v000)
        vx01 = v001 + npoint[0] * (v101 - v001)
        vx10 = v010 + npoint[0] * (v110 - v010)
        vx11 = v011 + npoint[0] * (v111 - v011)

        #reduce y
        vxy0 = vx00 + npoint[1] * (vx10 - vx00)
        vxy1 = vx01 + npoint[1] * (vx11 - vx01)

        #reduce z
        vxyz = vxy0 + npoint[2] * (vxy1 - vxy0)

        return vxyz

    def extract_anomalous_planes(self, axis, tol=1e-10):
        """ FUNCTION extract_anomalous_planes
        checks the average value of each plane against the average value of the cube file to find
        regions of interest
        """
        anomalous_planes = []
        axis_set = {0, 1, 2}
        axis_set.remove(axis)
        axis_list = list(axis_set)
        mean2D = np.mean(self.f_field, axis_list[0])
        mean1D = np.mean(mean2D, axis_list[1]-1)
        print(mean1D)
        mean0D = np.median(mean1D)
        #print(mean0D)
        for idx, mean in enumerate(mean1D):
            if mean0D < 1e-5:
                diff = abs(mean - mean0D)
            else:
                diff = abs(mean - mean0D) / mean0D
            #print(diff)
            if diff > tol:
                anomalous_planes.append(idx)
        print(anomalous_planes)

    def find_nonzero_lines(self, axis):
        nonzero_lines = []
        axis_set = {0, 1, 2}
        axis_set.remove(axis)
        axis_tup = tuple(axis_set)
        #axis_list = list(axis_set).append(axis)
        #field = np.transpose(self.f_field, axis_list)
        #for idx in range(field.shape[0]):
        #    for jdx in range(field.shape[1]):
        #        line = field[idx, jdx, :]
        #        if line.any 
        a = np.any(self.f_field, axis=axis_tup)
        print(a)

    def plotline(self, line, res=0, scale=1, transform=True, interpolate=False, save=False):
        """ FUNCTION plotline
        plots the field along the defined line

        INPUT:
        line:       line, the input line, defined along the cartesian basis
        integer:    res, the resolution (if 0 then assume gridsize)
        bool:       interpolate, a flag that determines whether trilinear interpolation should be used

        Shows a matplotlib output
        """
        if res == 0:
            res = self.f_size[np.argmax(line.dir)]
        # transform the position and direction to fit the grid, move position
        # to edge of grid
        if transform:
            tpos = self.trans_point_to_space(line.pos)
            tdir = self.trans_point_to_space(line.dir)
            tline = li.line(pos=tpos, dir=tdir)
        else:
            tline = line

        validCheck = self.has_line(tline)
        if not validCheck[0]:
            sys.exit("line choice invalid")
        start = validCheck[1][0]
        end = validCheck[1][1]
        # hacky way to avoid dups
        if np.array_equal(start, end):
            end = validCheck[1][2]

        if not tline.dir[0] == 0:
            if tline.dir[0] / (end[0] - start[0]) < 0:
                tstart = end
                tend = start
            else:
                tstart = start
                tend = end
        elif not tline.dir[1] == 0:
            if tline.dir[1] / (end[1] - start[1]) < 0:
                tstart = end
                tend = start
            else:
                tstart = start
                tend = end
        elif not tline.dir[2] == 0:
            if tline.dir[2] / (end[2] - start[2]) < 0:
                tstart = end
                tend = start
            else:
                tstart = start
                tend = end
        else:
            sys.exit("direction invalid")

        ndir = (tend - tstart) / res

        # pick integer values of position each time
        # ppos = parameterized position
        ppos = np.arange(res)
        data = np.zeros(res)
        for i in ppos:
            ipos = tstart + (ndir * i)
            if not interpolate:
                ipos = np.rint(ipos)
                try:
                    data[i] = self.f_field[int(ipos[0]), int(ipos[1]), int(ipos[2])]
                except IndexError:
                    # fix later
                    print("WARNING: plotpoint (%i, %i, %i) ignored" % (ipos[0], ipos[1], ipos[2]))
                    continue
            else:
                data[i] = self.interpolate(ipos)

        data = scale * data

        if save:
            f = plt.figure(figsize=(12, 1), dpi=133)
        plt.plot(ppos, data)
        #plt.show()
        if save:
            f.savefig('temp.png', transparent=True)
        return data

    def plotaverageline(self, axis):
        """ FUNCTION plotaverageline
        Plots the average along a principal direction (TODO, extend to arbitrary direction)

        INPUT:
        integer:    axis, the direction to plot
        """
        if axis == 0:
            data = np.average(np.average(self.f_field, axis=1), axis=1)
            plt.plot(np.arange(self.f_size[0]), data)
        if axis == 1:
            data = np.average(np.average(self.f_field, axis=0), axis=1)
            plt.plot(np.arange(self.f_size[1]), data)
        if axis == 2:
            data = np.average(np.average(self.f_field, axis=0), axis=0)
            plt.plot(np.arange(self.f_size[2]), data)
        #plt.show()
    
    def plotsimpleplane(self, axis, pos, save=""):
        fielddata = None
        if axis == 0:
            fielddata = self.f_field[pos,:,:]
        elif axis == 1:
            fielddata = self.f_field[:,pos,:]
        elif axis == 2:
            fielddata = self.f_field[:,:,pos]
        if save:
            img.imsave(save, arr=fielddata, cmap=cm.jet)
        else:
            plt.matshow(fielddata)
            plt.colorbar()
        return fielddata

    def plotnormalizedplane(self, axis, pos):
        """ FUNCTION plotsimpleplane
        Plots a plane defined by axis and pos, where axis is a principal direction

        INPUT:
        integer:    axis, the direction normal of plane to plot
        integer:    pos, the index of the line/plane intersection
        """
        plane_dim = np.delete(self.f_size, axis)
        X, Y = np.mgrid[-1:1:complex(0, plane_dim[0]), -1:1:complex(0, plane_dim[1])]
        if axis == 0:
            #plt.matshow(self.f_field[pos,:,:], cmap='hot')
            Z = self.f_field[pos,:,:]
        if axis == 1:
            #plt.matshow(self.f_field[:,pos,:], cmap='hot')
            Z = self.f_field[:,pos,:]
        if axis == 2:
            #plt.matshow(self.f_field[:,:,pos], cmap='hot')
            Z = self.f_field[:,:,pos]
        pcm = plt.pcolormesh(X, Y, Z, norm=nz.Normalize(midpoint=0.), cmap='RdBu_r')
        plt.colorbar(pcm, extend='both')
        #plt.colorbar()
        #plt.show()

    def rollfield(self):
        uroll = self.f_size // 2
        self.f_field = np.roll(self.f_field, uroll, axis=(0, 1, 2))

    def padfield(self):
        """ Pads and returns the field to fix for ipyvolume's show function
        """
        newsize = self.f_size + 16 - (self.f_size % 16)
        temp = np.zeros((newsize), dtype=float)
        temp[:self.f_size[0], :self.f_size[1], :self.f_size[2]] = self.f_field
        return temp

    def plotrolledplane(self, axis, pos, uroll=[0, 0]):
        """ FUNCTION plotrolledplane
        Plots a plane midway through a space defined by axis, rolled by uroll
        uroll expects a 2-list
        """
        plane_dim = np.delete(self.f_size, axis)
        uplane = np.zeros((plane_dim), dtype=float)
        if uroll[0] == 0 and uroll[1] == 0:
            print("x")
            uroll = [plane_dim[0] // 2, plane_dim[1] // 2]
        if axis == 0:
            uplane = np.roll(self.f_field[pos,:,:], uroll, axis=(0, 1))
        if axis == 1:
            uplane = np.roll(self.f_field[:,pos,:], uroll, axis=(0, 1))
        if axis == 2:
            uplane = np.roll(self.f_field[:,:,pos], uroll, axis=(0, 1))
        plt.matshow(uplane, cmap='hot')
        plt.colorbar()

    def plotaverageplane(self, axis):
        if axis == 0:
            plt.matshow(np.average(self.f_field, axis=0), cmap='hot')
        if axis == 1:
            plt.matshow(np.average(self.f_field, axis=1), cmap='hot')
        if axis == 2:
            plt.matshow(np.average(self.f_field, axis=2), cmap='hot')
        plt.colorbar()

    def plotplane(self, iplane, res=0, interpolate=False):
        t0 = self.trans_point_to_space(iplane.p)
        tu = self.trans_point_to_space(iplane.u)
        tv = self.trans_point_to_space(iplane.v)
        tn = self.trans_point_to_space(iplane.n)
        nu = ul.normalize(tu)
        nv = ul.normalize(tv)
        tplane = plane.plane(p=t0, n=tn)
        nplane = plane.plane(p=t0, u=nu, v=nv)
        print("plane equation:", "pos", t0, "u vec", tu, "v vec", tv, "normal", tn)

        # now have the transformed equation of the plane
        # can think of cube as planes with boundaries
        # step 0: normalize direction vectors
        if res == 0:
            res = self.f_size[np.argmax(tu)]

        # step 1: take each cube plane and find intersections between planes

        # face 1: xy plane, z = 0
        c1n = np.array([0, 0, 1])
        c1p = np.array([0, 0, 0])
        c2n = np.array([0, 0, 1])
        c2p = np.array([0, 0, self.f_size[2]])
        c3n = np.array([0, 1, 0])
        c3p = np.array([0, 0, 0])
        c4n = np.array([0, 1, 0])
        c4p = np.array([0, self.f_size[1], 0])
        c5n = np.array([1, 0, 0])
        c5p = np.array([0, 0, 0])
        c6n = np.array([1, 0, 0])
        c6p = np.array([self.f_size[0], 0, 0])

        face1 = plane.plane(p=c1p, n=c1n)
        face2 = plane.plane(p=c2p, n=c2n)
        face3 = plane.plane(p=c3p, n=c3n)
        face4 = plane.plane(p=c4p, n=c4n)
        face5 = plane.plane(p=c5p, n=c5n)
        face6 = plane.plane(p=c6p, n=c6n)

        l1 = ul.plane_intersection(face1, tplane)
        l2 = ul.plane_intersection(face2, tplane)
        l3 = ul.plane_intersection(face3, tplane)
        l4 = ul.plane_intersection(face4, tplane)
        l5 = ul.plane_intersection(face5, tplane)
        l6 = ul.plane_intersection(face6, tplane)

        # step 2: are these intersections in the boundaries?
        (l1valid, l1pts) = self.has_line(l1)
        #print("line 1: ", l1pos, l1dir, "validity:", l1valid, l1pts)
        (l2valid, l2pts) = self.has_line(l2)
        #print("line 2: ", l2pos, l2dir, "validity:", l2valid, l2pts)
        (l3valid, l3pts) = self.has_line(l3)
        #print("line 3: ", l3pos, l3dir, "validity:", l3valid, l3pts)
        (l4valid, l4pts) = self.has_line(l4)
        #print("line 4: ", l4pos, l4dir, "validity:", l4valid, l4pts)
        (l5valid, l5pts) = self.has_line(l5)
        #print("line 5: ", l5pos, l5dir, "validity:", l5valid, l5pts)
        (l6valid, l6pts) = self.has_line(l6)
        #print("line 6: ", l6pos, l6dir, "validity:", l6valid, l6pts)

        # step 3: get a square in the basis of the plane that encompasses the intersection completely
        intersection_list = l1pts + l2pts + l3pts + l4pts + l5pts + l6pts
        if not intersection_list:
            sys.exit("no intersection possible")
        is_np = np.array(intersection_list)
        #print("intersection list = ", is_np)
        is_u = np.zeros((len(intersection_list)), dtype=float)
        is_v = np.zeros((len(intersection_list)), dtype=float)
        for i, val in enumerate(is_np):
            (is_u[i], is_v[i]) = ul.get_jk(nplane, val)

        # step 4: find the range of constants in the equation of the plane (v form)
        ##print("is_u", is_u, "is_v", is_v)
        umin = np.amin(is_u)
        umax = np.amax(is_u)
        vmin = np.amin(is_v)
        vmax = np.amax(is_v)

        # step 5: loop through points and perform interpolation if necessary
        urange = umax - umin
        vrange = vmax - vmin
        if vrange >= urange:
            vinc = vrange / res
            uinc = vinc
        else:
            uinc = urange / res
            vinc = uinc
        vni = int(math.ceil(vrange / vinc))
        uni = int(math.ceil(urange / uinc))
        data = np.zeros((uni, vni), dtype=float)
        for iu in np.arange(uni):
            for iv in np.arange(vni):
                dpos = np.array([0.1, 0.1, 0.1])
                ipos = t0 + (nu * (umin + (iu * uinc))) + (nv * (vmin + (iv * vinc)))
                if not interpolate:
                    ipos = np.rint(ipos)
                    if self.has_point(ipos+dpos):
                        #print(ipos)
                        data[iu, iv] = self.field[int(ipos[0]), int(ipos[1]), int(ipos[2])]
                    else:
                        data[iu, iv] = np.nan
                else:
                    dpos = dpos * 11
                    if self.has_point(ipos+dpos):
                        data[iu, iv] = self.interpolate(ipos)
                    else:
                        data[iu, iv] = np.nan

        plt.matshow(data, cmap='hot')
        plt.colorbar()
        #plt.show()
        return data

    def clone(self, mode=0, verbosity=0):
        """ FUNCTION clone
        copies the basic size parameters of the field into a new field object
        if mode=1 then also copies the contents (TODO)

        INPUT:
        integer(opt): mode, flag to specify the nature of the clone
        """
        cfield = field()
        if mode == 0:
            np.copyto(cfield.f_size, self.f_size)
            np.copyto(cfield.f_orig, self.f_orig)
            np.copyto(cfield.f_t, self.f_t)
            if self.f_field is not None:
                cfield.f_field = np.zeros((cfield.f_size[0], cfield.f_size[1],
                                           cfield.f_size[2]), dtype=float)
            if verbosity > 0:
                print("created new field:")
                print("size=", cfield.f_size)
                print("transform=", cfield.f_t)
        return cfield

    def volume(self):
        # assume cell is a cube for now
        return np.sum(self.f_field) * np.prod(np.diag(self.f_t))

if __name__ == '__main__':
    pass
