import os
import sys
import re
import glob
import subprocess
import datetime
import time
import pytz
import argparse

import ee
import ee.cli
import ee.cli.commands
import ee.cli.utils

input_dir = '../data/vaklodingen_asc/'
output_dir = '../data/vaklodingen_tif/'

os.chdir(input_dir)

files = glob.glob('./*.asc')

print(files)

local = pytz.timezone("Europe/Amsterdam")


# path -> pathlib
# readlines() -> for line in file ... break
# datetime -> dateutil

ee.Initialize()
ee_config = ee.cli.utils.CommandLineConfig()


# disable output buffering (monitor output)
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def run(cmd):
    print(cmd)
    subprocess.call(cmd)

start_index = 0

for i, f in enumerate(files[start_index:]):
    print('Processing file ' + f + ', file index: ' + str(i + start_index))

    # extract time in UTC
    datestring = re.findall(r"\d{8}", f)[0]
    t = datetime.datetime.strptime(datestring, '%Y%m%d')
    local_t = local.localize(t, is_dst=None)
    utc_t = local_t.astimezone(pytz.utc)
    time_start = utc_t.strftime('%Y-%m-%d')

    # parse file names
    filename = os.path.basename(f)
    filename_no_ext = os.path.splitext(filename)[0]
    filename_output = '../' + output_dir + '/{0}.tif'.format(filename_no_ext)

    # get nodata value ... UGLY, UGLY code!
    nodata_value = -99999999
    with open(f) as asc:
         for line in asc:
             if "nodata_value" in line.lower():
                 nodata_value = line.split()[1]
                 break

    # convert to TIF
    run("gdal_translate -ot Float32 -a_srs EPSG:28992 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=6 -of GTiff {0} {1}".format(filename, filename_output))

    # copy to GCS 
    run(r"gsutil.cmd cp {0} gs://hydro-earth/vaklodingen/{1}".format('../' + output_dir + '/{0}.tif'.format(filename_no_ext), filename_no_ext))

    # upload to GEE
    retry_count = 0

    while True:
      run("earthengine upload image --wait --asset_id=users/gena/vaklodingen/{1} --nodata_value={0} gs://hydro-earth/vaklodingen/{1}".format(nodata_value, filename_no_ext))

      # check last task status
      tasks = ee.data.getTaskList()
      task_state = None
      for task in tasks:
          task_status = ee.data.getTaskStatus([task['id']])
          task_state = task_status[0]['state']
          print(task_status)
          break
      
      if task_state != 'FAILED':
        break # done
      else:
        retry_count += 1
        print('Retrying upload ' + str(retry_count) + ' ...')

        if retry_count > 10:
            print('Maximum number of retry reached, exiting ...')
            sys.exit(0)

    # set time
    run("earthengine asset set --time_start {0} users/gena/vaklodingen/{1}".format(time_start, filename_no_ext))


    # upload
    #  parser = argparse.ArgumentParser()
    #  cmd_upload = ee.cli.commands.UploadCommand(parser)
    #  args = parser.parse_args([
    #      'image', 
    #      '--nodata_value', nodata_value, 
    #      '--asset_id', 'users/gena/vaklodingen/' + filename_no_ext, 
    #      'gs://hydro-earth/vaklodingen/' + filename_no_ext#,
    #      '--wait'
    #  ])
    #
    #  cmd_upload.run(args, ee_config)
   

    # ls
    # parser = argparse.ArgumentParser()
    # cmd_list = ee.cli.commands.ListCommand(parser)
    # args = parser.parse_args(['users/gena/vaklodingen'])
    # cmd_list.run(args, ee_config)

    # make public - not needed, parent is a collection
    # run("earthengine acl set public users/gena/vaklodingen/{0}".format(filename_no_ext))




