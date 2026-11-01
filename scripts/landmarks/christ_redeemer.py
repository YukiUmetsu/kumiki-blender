"""Procedural Christ the Redeemer silhouette study for Blender 4.2+.

Run: blender -b --python christ_redeemer.py -- --output /tmp/christ.glb
This is an original geometric study, not a scan or a likeness-certified replica.
"""
import bpy, math, sys
from mathutils import Vector

def mat(name, color):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); return m

STONE=mat("Soapstone mosaic - warm grey",(0.61,0.61,0.57))
SHADOW=mat("Recessed carving",(0.39,0.40,0.38))

def mesh(name, rings, material=STONE):
    n=len(rings[0]); verts=[v for ring in rings for v in ring]
    faces=[]
    for j in range(len(rings)-1):
        for i in range(n):
            a=j*n+i;b=j*n+(i+1)%n
            faces.append((a,b,b+n,a+n))
    faces += [tuple(reversed(tuple(range(n)))),tuple((len(rings)-1)*n+i for i in range(n))]
    me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update()
    ob=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    for p in me.polygons:p.use_smooth=True
    return ob

def ellipsoid(name,loc,scale,material=STONE,segments=48,rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,location=loc)
    o=bpy.context.object;o.name=name;o.scale=scale;o.data.materials.append(material)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    for p in o.data.polygons:p.use_smooth=True
    return o

def torso():
    rings=[]
    for z,rx,ry in [(3.0,1.02,.53),(4.3,1.13,.62),(6.1,1.26,.66),(8.0,1.58,.73),(9.5,1.93,.82),(10.3,2.04,.83)]:
        rings.append([(rx*math.cos(t),ry*math.sin(t)-.05,z) for t in [2*math.pi*i/80 for i in range(80)]])
    mesh("Robe | taper and shoulder mass",rings)
    for side in (-1,1):
        # Draped sleeve is a thick volume, with open-ended flared cuffs.
        sections=[]
        for x,halfwidth,depth,z in [(side*1.55,.72,.55,9.45),(side*3.2,.86,.54,9.25),(side*5.1,.70,.42,8.99),(side*7.1,.44,.30,8.93),(side*8.55,.31,.25,9.0)]:
            sections.append([(x,depth*math.sin(t),z+halfwidth*math.cos(t)) for t in [2*math.pi*i/48 for i in range(48)]])
        mesh("Sleeve | "+str(side),sections)
        # Hands project beyond the sleeve, palms facing upward.
        ellipsoid("Hand | palm "+str(side),(side*9.1,0,9.02),(.66,.25,.12))
        for finger in range(4):
            ellipsoid("Finger "+str(side)+" "+str(finger),(side*(9.53+.11*math.sin(finger)),(finger-1.5)*.105,9.02),(.39,.055,.055),segments=16,rings=10)
        ellipsoid("Thumb "+str(side),(side*8.96,-.31,8.96),(.30,.09,.08))
        # Raised carved folds along underside of each sleeve.
        for k in range(4):
            x=side*(2.3+k*1.28)
            ellipsoid("Sleeve fold "+str(side)+" "+str(k),(x,-.44,8.90-.07*k),(.08,.13,.47-.065*k),SHADOW,24,12)
    # Vertical long stole and robe panels.
    for side in (-1,1):
        rings=[]
        for z in [3.1,4.2,5.5,7.0,8.6,10.0]:
            x=side*(.67+.11*math.sin(z*.8))
            rings.append([(x+u,-.68+.06*math.sin(z),z) for u in [-.22,-.16,.16,.22]])
        # narrow cloth strips via closed rectangular rings
        sections=[]
        for z in [3.1,4.2,5.5,7,8.6,10]:
            x=side*(.67+.11*math.sin(z*.8))
            sections.append([(x-.17,-.78,z),(x+.17,-.78,z),(x+.17,-.69,z),(x-.17,-.69,z)])
        mesh("Stole "+str(side),sections)
    for k in range(14):
        x=-.93+k*.14
        rings=[]
        for z in [3.05,3.6,4.5,5.3]:
            y=-.52-.05*math.cos(k*.8)
            rings.append([(x-.026,y,z),(x+.026,y,z),(x+.026,y+.045,z),(x-.026,y+.045,z)])
        mesh("Lower robe pleat %02d"%k,rings,SHADOW if k%3==0 else STONE)

def head():
    # Front is negative Y; skull narrowed toward chin, prominent forehead and jaw.
    levels=[(10.3,.40,.35),(10.65,.53,.45),(11.0,.69,.48),(11.5,.75,.53),(12.0,.76,.56),(12.45,.68,.53),(12.8,.49,.40),(12.98,.16,.21)]
    rings=[]
    for z,rx,ry in levels:
        rings.append([(rx*math.cos(t),-.06+ry*math.sin(t),z) for t in [2*math.pi*i/80 for i in range(80)]])
    mesh("Face | sculpted skull",rings)
    # Hair falls on both sides, not a hood attached to the cheeks.
    for side in (-1,1):
        for j in range(3):
            ellipsoid("Hair strand %s %s"%(side,j),(side*(.68+j*.055),.11+j*.06,11.6-j*.16),(.14,.36,1.40-j*.09),STONE,32,20)
    for side in (-1,1):
        ellipsoid("Brow "+str(side),(side*.35,-.52,12.15),(.32,.11,.075),SHADOW)
        ellipsoid("Upper eyelid "+str(side),(side*.34,-.536,12.04),(.29,.10,.065),STONE)
        ellipsoid("Eye socket "+str(side),(side*.34,-.548,11.98),(.24,.045,.065),SHADOW)
        ellipsoid("Cheek "+str(side),(side*.43,-.43,11.56),(.28,.13,.29))
    ellipsoid("Nasal bridge",(0,-.54,11.93),(.14,.20,.40))
    ellipsoid("Nose tip",(0,-.74,11.65),(.20,.16,.13))
    for side in (-1,1):
        ellipsoid("Nostril wing "+str(side),(side*.16,-.68,11.59),(.09,.09,.065))
    ellipsoid("Upper lip",(0,-.54,11.37),(.29,.09,.065))
    ellipsoid("Lip crease",(0,-.618,11.33),(.28,.028,.022),SHADOW)
    ellipsoid("Lower lip",(0,-.54,11.27),(.27,.09,.06))
    ellipsoid("Chin",(0,-.38,10.94),(.42,.18,.24))
    ellipsoid("Neck",(0,.0,10.42),(.36,.32,.55))
    # Central parted hair contour at crown.
    for side in (-1,1):
        ellipsoid("Hair crown "+str(side),(side*.34,.13,12.74),(.42,.37,.32))

def add_surface_details():
    """Sculptural secondary forms: flowing chest drape, robe hems and hair grooves."""
    for i in range(9):
        x=-1.05+i*.255
        z=7.7-.28*math.cos(i*.47)
        ellipsoid("Chest drapery ridge %02d"%i,(x,-.715,z),(.12,.055,.36),STONE,24,12)
    for i in range(7):
        z=6.6+i*.42
        ellipsoid("Diagonal chest fold %02d"%i,(-.25+i*.10,-.73,z),(.72-i*.055,.055,.065),STONE,32,12)
    for side in (-1,1):
        for i in range(5):
            z=11.0+i*.32
            ellipsoid("Hair carved groove %s %s"%(side,i),(side*(.77-.027*i),-.16,z),(.035,.055,.30),SHADOW,20,12)
    for i in range(8):
        x=-.72+i*.20
        ellipsoid("Vertical robe channel %02d"%i,(x,-.55,4.65),(.028,.043,1.35),SHADOW,20,12)

def make_camera(name, location, target, ortho):
    bpy.ops.object.camera_add(location=location)
    cam=bpy.context.object;cam.name=name
    direction=Vector(target)-cam.location
    cam.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
    cam.data.type='ORTHO';cam.data.ortho_scale=ortho
    return cam

def inspection_views(output_dir):
    """Render orthographic front/profile and head crops for human review."""
    import os
    os.makedirs(output_dir,exist_ok=True)
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE_NEXT'
    scene.render.resolution_x=900;scene.render.resolution_y=1100
    scene.render.resolution_percentage=100
    scene.world.color=(.17,.19,.22)
    bpy.ops.object.light_add(type='AREA', location=(4,-9,17))
    light=bpy.context.object;light.name='QA softbox';light.data.energy=2300;light.data.shape='DISK';light.data.size=9
    light.rotation_euler=(Vector((0,0,10))-light.location).to_track_quat('-Z','Y').to_euler()
    for name,loc,target,scale in [
        ('front',(0,-38,8),(0,0,7.0),22.0),
        ('profile',(38,0,8),(0,0,7.0),22.0),
        ('head_front',(0,-18,11.7),(0,0,11.65),3.5),
        ('head_profile',(18,0,11.7),(0,0,11.65),3.5),
    ]:
        cam=make_camera('QA camera '+name,loc,target,scale)
        scene.camera=cam
        scene.render.filepath=os.path.join(output_dir,name+'.png')
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam,do_unlink=True)

def main():
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    torso();head();add_surface_details()
    # Flat octagonal pedestal, kept separate from statue.
    bpy.ops.mesh.primitive_cylinder_add(vertices=8,radius=2.4,depth=2.7,location=(0,0,1.35))
    o=bpy.context.object;o.name="Pedestal";o.data.materials.append(mat("Dark granite",(.25,.26,.25)))
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    if '--views' in args:
        inspection_views(args[args.index('--views')+1])
    if '--output' in args:
        path=args[args.index('--output')+1]
        bpy.ops.export_scene.gltf(filepath=path,export_format='GLB')
    if '--blend' in args:
        bpy.ops.wm.save_as_mainfile(filepath=args[args.index('--blend')+1])
if __name__=='__main__':main()
