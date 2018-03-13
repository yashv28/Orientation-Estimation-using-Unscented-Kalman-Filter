from scipy import io
import numpy as np
import os
from quat_helper import *
import matplotlib.pyplot as plt
import cv2
from math import pi
#from skvideo.io import FFmpegWriter
import cPickle as pickle

#######################################################################
#######################################################################

Dataset = 1
est = False#True #Set False for stitching with vicon data
stitch = True #Set False for skipping panorama

#######################################################################
#######################################################################

def cylindrical_projection(src, rot, dst=None):
	nRows, nCols , c = src.shape
	y = np.linspace(0, nCols - 1, nCols)
	z = np.linspace(0, nRows - 1, nRows)
	yy, zz = np.meshgrid(y, z)
	yy = yy.reshape((1, nCols*nRows))
	zz = zz.reshape((1, nCols*nRows))
	rgb = np.transpose(src, (2, 0, 1)).reshape((3, nCols*nRows)).T

	yy = (nCols / 2) - yy
	zz = (nRows / 2) - zz

	hFOV = 60 * (pi/180)
	yFactor = hFOV / nCols
	vFOV = 45 * (pi/180)
	zFactor = vFOV / nRows
	longitude = yy * yFactor
	latitude = zz * zFactor

	rho = 1
	X = rho * np.cos(latitude) * np.cos(longitude)
	Y = rho * np.cos(latitude) * np.sin(longitude)
	Z = rho * np.sin(latitude)
	XYZ = np.vstack((X, Y, Z))

	wXYZ = rot.dot(XYZ)

	wX = wXYZ[0, :]
	wY = wXYZ[1, :]
	wZ = wXYZ[2, :]
	longitude = np.arctan2(wY, wX)
	latitude = np.arctan2(wZ, np.sqrt(wX**2 + wY**2))

	cylHeight = ((1 / zFactor) * (-np.tan(latitude) + pi/2)).astype(np.uint32)
	cylAngle = ((1 / yFactor) * (-longitude + pi)).astype(np.uint32)

	if dst is None:
		dst = np.zeros((int(pi / zFactor), int(2 * pi / yFactor), 3)).astype(np.uint8)

	try:
		dst[cylHeight, cylAngle, :] = rgb

	except IndexError:
		cylHeight[cylHeight > dst.shape[0] - 1] = dst.shape[0] - 1
		dst[cylHeight, cylAngle, :] = rgb

	return dst

def gaussian_update(qt, ut, P, Q):   
	qu = vec2quat(ut)

	tmp=np.matrix(np.zeros([4,6]))
	L = np.linalg.cholesky(P+Q)
	n,m = np.shape(P)
	left_vec = L*np.sqrt(2*n)
	right_vec = -L*np.sqrt(2*n)
	new_vec = np.hstack((left_vec, right_vec))
	nr,nc = np.shape(new_vec)
	
	v = np.matrix(np.zeros([3,6]))
	for i in range(0,nc):
		temp = vec2quat(new_vec[:,i])
		tmp[:,i] = np.transpose(multiply_quaternions(temp,qt))

	sigma_points = np.transpose(tmp)
	motion_sig = np.zeros(np.shape(sigma_points))
	for i in range(0,6):
		motion_sig[i] = multiply_quaternions(sigma_points[i], qu)
	next_qt, error = quat_average(motion_sig, qt)
	nr,nc = np.shape(error)
	next_cov = np.zeros([nc,nc])
	for i in range(0,nr):
		temp_cov = np.transpose(error[i])*error[i]
		next_cov += temp_cov

	next_cov=next_cov/12
	return next_qt, next_cov, sigma_points, error

	
def sigma_update(sigma_points, g, R):
	new_sigma = np.zeros(np.shape(sigma_points))
	z = np.zeros([np.shape(sigma_points)[0]-1,np.shape(sigma_points)[1]])
	for i in range(0,np.shape(sigma_points)[0]):
		new_sigma[i] = multiply_quaternions(multiply_quaternions(inverse_quaternion(sigma_points[i]),g),sigma_points[i])
		
	z = new_sigma[:,1:]
	z_mean = np.mean(z,0)
	return z, z_mean
	
	
def calcpzz(z, z_mean):
	temp = np.matrix(z - z_mean)
	pzz = np.zeros([np.shape(z)[1], np.shape(z)[1]])
	for i in range(0,np.shape(temp)[0]):
		pzz_temp = np.transpose(temp[i])*temp[i]
		pzz += pzz_temp
		
	return pzz/12.0
	
	
def calcpxz(error, z, z_mean):    
	temp = np.matrix(z - z_mean)
	pxz = np.zeros([np.shape(z)[1], np.shape(z)[1]])
	for i in range(0,np.shape(error)[0]):
		pxz_temp = np.transpose(error[i])*temp[i]
		pxz += pxz_temp
		
	return pxz/12.0


########################################################################
########################################################################

#Data Load and timestamp match

imu = io.loadmat("imu/imuRaw"+str(Dataset)+".mat")
imu_vals = imu['vals']
imu_vals = np.transpose(imu_vals)
imu_ts = imu['ts']
yts = imu_ts
imu_ts = np.transpose(imu_ts)

Vref = 3300

acc_x = -np.array(imu_vals[:,0])
acc_y = -np.array(imu_vals[:,1])
acc_z = np.array(imu_vals[:,2])
acc = [acc_x, acc_y, acc_z]

acc = np.array(acc)
acc = np.transpose(acc)
acc_sensitivity = 330.0
acc_scale_factor = Vref/1023.0/acc_sensitivity
acc_bias = acc[0] - (np.array([0, 0, 1])/acc_scale_factor)
acc_val = acc*acc_scale_factor
acc_val = acc_val - (acc_bias)*acc_scale_factor

gyro_x = np.array(imu_vals[:,4])
gyro_y = np.array(imu_vals[:,5])
gyro_z = np.array(imu_vals[:,3])

gyro = [gyro_x, gyro_y, gyro_z]
gyro = np.array(gyro)
gyro = np.transpose(gyro)
gyro_bias = gyro[0]
gyro_sensitivity = 3.33
gyro_scale_factor = Vref/1023/gyro_sensitivity
gyro_val = gyro*gyro_scale_factor
gyro_val = (np.array(gyro_val) - (gyro_bias*gyro_scale_factor))*(np.pi/180)

if os.path.exists("vicon/viconRot"+str(Dataset)+".mat"):

	vicon = io.loadmat("vicon/viconRot"+str(Dataset)+".mat")

	vicon_vals = vicon['rots']
	vicon_ts = vicon['ts']

	vicon_phi = np.zeros([np.shape(vicon_vals)[2], 1])
	vicon_theta = np.zeros([np.shape(vicon_vals)[2], 1])
	vicon_psi = np.zeros([np.shape(vicon_vals)[2], 1])
	for i in range(np.shape(vicon_vals)[2]):
		R = vicon_vals[:,:,i]
		vicon_phi[i], vicon_theta[i], vicon_psi[i] = rot2euler(R)

else:
	est = True

########################################################################
########################################################################  
	
P = 0.00001*np.identity(3)
Q = 0.00001*np.identity(3)
R = 0.0001*np.identity(3)
q0 = np.matrix([1, 0, 0, 0])
qt = np.matrix([1, 0, 0, 0])
ut = gyro_val[0]
g = np.matrix([0, 0, 0, 1])
t = imu_ts.shape[0]
R_calc = np.zeros((3, 3, np.shape(gyro_val)[0]))

#UKF
if not os.path.exists('Parameters/param'+str(Dataset)+'.pickle'):
	for i in range(0,np.shape(gyro_val)[0]):
		
		if i==0:
			ut = gyro_val[i]*imu_ts[0]
			predicted_q = q0
		else:
			ut = gyro_val[i]*(imu_ts[i] - imu_ts[i-1])
		
		next_q, next_cov, sigma_points, error = gaussian_update(qt, ut, P, Q)

		z, z_mean = sigma_update(sigma_points, g, R)
		z = np.matrix(z)
		z_mean = np.matrix(z_mean)

		pzz = calcpzz(z, z_mean)
		pvv = pzz + R
		pxz = calcpxz(error, z, z_mean)

		K = np.dot(pxz,np.linalg.inv(pvv))

		I = np.transpose(acc_val[i] - z_mean)
		KI = vec2quat(np.transpose(K*I))
		qt = np.matrix(np.empty([1,4]))
		qt = multiply_quaternions(KI,next_q)
		P = next_cov - np.dot(np.dot(K,pvv),np.transpose(K))

		predicted_q = np.vstack((predicted_q, qt))

		R_calc[:, :, i] = quat2rot(qt)

	with open('Parameters/param'+str(Dataset)+'.pickle', "wb") as f:
		pickle.dump((predicted_q, R_calc), f)

else:
	with open('Parameters/param'+str(Dataset)+'.pickle', "rb") as f:
		predicted_q, R_calc = pickle.load(f) 

phi = np.zeros([np.shape(predicted_q)[0], 1])
theta = np.zeros([np.shape(predicted_q)[0], 1])
psi = np.zeros([np.shape(predicted_q)[0], 1])

for i in range(np.shape(predicted_q)[0]):
	R = quat2rot(predicted_q[i])
	phi[i], theta[i], psi[i] = rot2euler(R)
  

if os.path.exists("vicon/viconRot"+str(Dataset)+".mat"):
	plt.figure(1)
	plt.subplot(311)
	plt.plot(vicon_phi, 'b', phi, 'r')
	plt.ylabel('Roll')
	plt.subplot(312)
	plt.plot(vicon_theta, 'b', theta, 'r')
	plt.ylabel('Pitch')
	plt.subplot(313)
	plt.plot(vicon_psi, 'b', psi, 'r')
	plt.ylabel('Yaw')
	plt.savefig('Results/RPY'+str(Dataset)+'.png')
	plt.show()

else:
	plt.figure(1)
	plt.subplot(311)
	plt.plot(phi, 'r')
	plt.ylabel('Roll')
	plt.subplot(312)
	plt.plot(theta, 'r')
	plt.ylabel('Pitch')
	plt.subplot(313)
	plt.plot(psi, 'r')
	plt.ylabel('Yaw')
	plt.savefig('Results/RPYwoVicon'+str(Dataset)+'.png')
	plt.show()

if stitch:

	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	if est:
		out = cv2.VideoWriter('Results/dataset_' + str(Dataset) + 'IMU.avi', fourcc, fps=24.0, frameSize=(1920, 960), isColor=True)
		#writer = FFmpegWriter('Results/dataset_' + str(Dataset) + 'IMU.avi')
	else:
		out = cv2.VideoWriter('Results/dataset_' + str(Dataset) + 'Vicon.avi', fourcc, fps=24.0, frameSize=(1920, 960), isColor=True)
		#writer = FFmpegWriter('Results/dataset_' + str(Dataset) + 'Vicon.avi')

	cam_Packet = io.loadmat("cam/cam"+str(Dataset)+".mat")
	cam_ImArrays = cam_Packet['cam']
	cam_ts = cam_Packet['ts']

	for im in range(cam_ImArrays.shape[3]):
		image = cam_ImArrays[:, :, :, im]

		if est:
			idx = np.where(np.isclose(yts[0], cam_ts[0, im], rtol=0, atol=1E-2))[0]
			if idx.size > 1:
				idx = idx[0]

			try:
				R = R_calc[:, :, idx].reshape((3, 3))
			except ValueError:
				break

		else:
			idx = np.where(np.isclose(vicon_ts[0], cam_ts[0, im], rtol=0, atol=1E-2))[0]
			if idx.size > 1:
				idx = idx[0]

			try:
				R = vicon_vals[:, :, idx].reshape((3, 3))
			except ValueError:
				break

		if im == 0:
			pano = cylindrical_projection(src=image, rot=R)
		else:
			pano = cylindrical_projection(src=image, rot=R, dst=pano)

		cv2.namedWindow('Stitching', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Stiching', pano.shape[0], pano.shape[1])
		cv2.imshow('Stitching', pano)
		out.write(pano)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		#writer.writeFrame(pano)

	#writer.close()
	out.release()
	cv2.destroyAllWindows()

	cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Panorama', pano.shape[0], pano.shape[1])
	cv2.imshow('Panorama', pano)
	if est:
		cv2.imwrite('Results/Pano'+str(Dataset)+'IMU.jpg',pano)
	else:
		cv2.imwrite('Results/Pano'+str(Dataset)+'Vicon.jpg',pano)
	cv2.waitKey(0)
