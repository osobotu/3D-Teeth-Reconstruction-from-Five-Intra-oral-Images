import copy
import glob
import os

import h5py
import numpy as np
import open3d as o3d
import skimage
from scipy.spatial.transform import Rotation as RR
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

from const import *

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200

def generateProjectedMeshImg_offscreen(
    renderer,
    ulTeethMshes,
    phType,
    ex_rxyz,
    ex_txyz,
    fx,
    u0,
    v0,
    rela_R,
    rela_t,
    rh,
    rw,
):
    ph = phType.value
    if phType in [PHOTO.UPPER, PHOTO.LOWER]:
        msh = copy.deepcopy(ulTeethMshes[ph])
    else:
        uMsh = copy.deepcopy(ulTeethMshes[0])
        lMsh = copy.deepcopy(ulTeethMshes[1])
        lMsh.rotate(rela_R.T, center=(0, 0, 0))
        lMsh.translate(rela_t)
        msh = uMsh + lMsh

    rotMat = RR.from_rotvec(ex_rxyz[ph]).as_matrix()
    msh.rotate(rotMat, center=(0, 0, 0))
    msh.translate(ex_txyz[ph])
    msh.compute_vertex_normals()

    scene = renderer.scene
    scene.clear_geometry()
    mat = MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", msh, mat)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        WINDOW_WIDTH, WINDOW_HEIGHT, fx[ph], fx[ph], u0[ph], v0[ph]
    )
    extrinsic = np.identity(4)
    renderer.setup_camera(intrinsic, extrinsic)

    img = renderer.render_to_image()
    img_np = np.asarray(img) / 255.0
    _u0 = WINDOW_WIDTH / 2 - 0.5
    _v0 = WINDOW_HEIGHT / 2 - 0.5
    tForm = skimage.transform.EuclideanTransform(
        translation=np.array([_u0 - u0[ph], _v0 - v0[ph]]),
        dimensionality=2,
    )
    shiftedImg = skimage.transform.warp(img_np, tForm)
    croppedImg = shiftedImg[:rh, :rw]
    return croppedImg

def readCameraParamsFromH5(h5File):
    with h5py.File(h5File, "r") as f:
        grp = f["EMOPT"]
        return (
            grp["EX_RXYZ"][:], grp["EX_TXYZ"][:], grp["FOCLTH"][:], grp["DPIX"][:],
            grp["U0"][:], grp["V0"][:], grp["RELA_R"][:], grp["RELA_T"][:]
        )

def meshProjection(renderer, tag):
    demoH5File = os.path.join(DEMO_H5_DIR, f"demo-tag={tag}.h5")
    upperTeethObj = os.path.join(DEMO_MESH_DIR, str(tag), f"Pred_Upper_Mesh_Tag={tag}.obj")
    lowerTeethObj = os.path.join(DEMO_MESH_DIR, str(tag), f"Pred_Lower_Mesh_Tag={tag}.obj")

    photos = []
    for phtype in PHOTO_TYPES:
        imgfile = glob.glob(os.path.join(PHOTO_DIR, f"{tag}-{phtype.value}.*"))[0]
        img = skimage.io.imread(imgfile)
        h, w = img.shape[:2]
        scale = RECONS_IMG_WIDTH / w
        rimg = skimage.transform.resize(img, (int(h * scale), RECONS_IMG_WIDTH, 3))
        photos.append(rimg)

    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = readCameraParamsFromH5(demoH5File)
    fx = focLth / dpix
    _color = [0.55, 0.7, 0.85]
    _alpha = 0.45

    upperTeethO3dMsh = o3d.io.read_triangle_mesh(upperTeethObj)
    upperTeethO3dMsh.paint_uniform_color(_color)
    upperTeethO3dMsh.compute_vertex_normals()

    lowerTeethO3dMsh = o3d.io.read_triangle_mesh(lowerTeethObj)
    lowerTeethO3dMsh.paint_uniform_color(_color)
    lowerTeethO3dMsh.compute_vertex_normals()

    for phType, img in zip(PHOTO_TYPES, photos):
        mshImg = generateProjectedMeshImg_offscreen(
            renderer,
            [upperTeethO3dMsh, lowerTeethO3dMsh],
            phType,
            ex_rxyz,
            ex_txyz,
            fx,
            u0,
            v0,
            rela_R,
            rela_t,
            img.shape[0],
            img.shape[1],
        )
        mesh_img_file = os.path.join(VIS_DIR, f"[new]-mesh-tag={tag}-{phType}.png")
        skimage.io.imsave(mesh_img_file, skimage.img_as_ubyte(mshImg))

        bkgrd = np.all(mshImg < 0.01, axis=-1)
        _teethRegion = np.tile(~bkgrd[..., None], (1, 1, 3))
        img = img[..., :3]
        np.putmask(
            img, _teethRegion, np.clip(_alpha * mshImg + (1.0 - _alpha) * img, 0.0, 1.0)
        )
        output = img
        output_img_file = os.path.join(VIS_DIR, f"[new]-overlay-tag={tag}-{phType}.png")
        skimage.io.imsave(output_img_file, skimage.img_as_ubyte(output))

def main(tag="0"):
    renderer = OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
    meshProjection(renderer, tag)

if __name__ == "__main__":
    main(tag="0")
