"""Render four neutral multi-view GLB head images without Blender.

Optional deps: trimesh, vtk (9.2+). Linux headless: xvfb-run -a python
scripts/render_glb_head_qa.py model.glb --output .review/head-qa

This is a fast silhouette / mesh-intersection gate. It does NOT replace
Blender Cycles/PBR QA, scan registration, or reference camera calibration.
"""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--size',type=int,default=850)
    parser.add_argument('--prefix',action='append',default=None,
                        help='Geometry name prefix to render; repeat. Default: Head |, Hair |, Face |')
    args=parser.parse_args()
    if args.size<128 or args.size>4096:
        parser.error('size must be between 128 and 4096')
    try:
        import trimesh
        import numpy as np
        import vtk
    except ImportError as error:
        parser.error('Install optional trimesh and vtk packages: '+str(error))
    scene=trimesh.load(args.model,force='scene')
    prefixes=tuple(args.prefix or ('Head |','Hair |','Face |'))
    meshes=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        if not name.startswith(prefixes):
            continue
        mesh=scene.geometry[name].copy()
        mesh.apply_transform(transform)
        if len(mesh.faces):
            meshes.append((name,mesh))
    if not meshes:
        raise ValueError('No head geometry matches prefixes '+repr(prefixes))
    bounds=np.vstack([m.bounds for _,m in meshes])
    low=bounds.min(axis=0);high=bounds.max(axis=0)
    center=(low+high)/2
    extent=float(max(high-low))
    if not np.isfinite(extent) or extent<=0:
        raise ValueError('Invalid GLB geometry bounds')
    actors=[]
    for name,mesh in meshes:
        points=vtk.vtkPoints()
        points.SetDataTypeToFloat()
        for vertex in mesh.vertices:
            points.InsertNextPoint(*vertex)
        triangles=vtk.vtkCellArray()
        for face in mesh.faces:
            tri=vtk.vtkTriangle()
            for i,k in enumerate(face):
                tri.GetPointIds().SetId(i,int(k))
            triangles.InsertNextCell(tri)
        poly=vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(triangles)
        normals=vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SetFeatureAngle(65)
        normals.SplittingOff()
        normals.ConsistencyOn()
        mapper=vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        actor=vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*((.65,.68,.66) if name.startswith('Hair |') else (.77,.77,.75)))
        actor.GetProperty().SetInterpolationToPBR()
        actor.GetProperty().SetMetallic(.02)
        actor.GetProperty().SetRoughness(.86)
        actors.append(actor)
    x,y,z=center
    views={
        'front':((x,y,z+3*extent),(x,y,z)),
        'low':((x,y-.36*extent,z+3*extent),(x,y,z)),
        'profile':((x+3*extent,y,z),(x,y,z)),
        'oblique':((x+2.3*extent,y+.12*extent,z+2.3*extent),(x,y,z)),
    }
    args.output.mkdir(parents=True,exist_ok=True)
    for name,(position,target) in views.items():
        renderer=vtk.vtkRenderer()
        renderer.SetBackground(.12,.14,.17)
        for actor in actors:
            renderer.AddActor(actor)
        for light_position,intensity in [((x+extent,y+extent,z+extent),1.1),
                                          ((x-extent,y-.5*extent,z+extent),.65)]:
            light=vtk.vtkLight()
            light.SetPosition(*light_position)
            light.SetFocalPoint(*center)
            light.SetIntensity(intensity)
            renderer.AddLight(light)
        window=vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.AddRenderer(renderer)
        window.SetSize(args.size,args.size)
        camera=renderer.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*target)
        camera.SetViewUp(0,1,0)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(extent*.66)
        renderer.ResetCameraClippingRange()
        window.Render()
        capture=vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.Update()
        writer=vtk.vtkPNGWriter()
        writer.SetFileName(str(args.output/(name+'.png')))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        window.Finalize()
    print('HEAD_GLB_QA',{'meshes':len(meshes),'extent_m':round(extent,3),
                         'output':str(args.output)})


if __name__=='__main__':
    main()
