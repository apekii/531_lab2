from Functions import *

# ************************* ENGO 531 LAB 2 *************************

# __________________________________________________________________

# ****************** Read In All Necessary Files *******************
# pho file, observation points
pho_LH = csv2mat('lab1_pho.pho')

# iop file
iop = csv2mat('lab1_int.int')

# eop file
eop = csv2mat('lab1_ext.ext')

# object points file, from D2L
obj = csv2mat('lab1_obj.txt')

# constants file (ie. configuration data)
# 1st row: image size in pixels (x, y)
# 2nd row: pixel spacing in micrometers (x, y)
# 3rd row: normal principal distance in mm and a '0' as a placeholder
# 4th row: std for image point coordinates, and std for control point coordinates
# 5th row: convergence criteria
constants = csv2mat('lab1_constants.txt')

# __________________________________________________________________

# ************************** Conversions ***************************
# turn pho from LH to RH; pho should already be sorted by imageID
pho_RH = lh2rh(pho_LH, constants)

# sort pho_RH by imageID
pho_RH.sort(key=lambda x: x[1])

# make large matrix with all necessary indices
indmat = indexIDmat(pho_RH, obj, eop, iop)

# calculate proper matrix sizes
nur = calc_nur(pho_RH, eop, obj)

# __________________________________________________________________

# ********************* Least Squares Prep *************************
# # Standard Deviation + Weight Matrices ***************************
# for image point coordinates, converted to mm
sigma2_i = (constants[3][0] * (constants[1][0] / 1000)) ** 2
P_i = weight(nur[0], sigma2_i)

# for control point coordinates
sigma2_e = constants[3][1] ** 2
P_e = weight(nur[0], sigma2_e)

# # P_o ************************************************************
# create matrix of 0's for
P_o = [[0 for i in range(0, nur[2])] for j in range(0, nur[2])]

# apply weight matrix if control point/GCP, which will be for the bottom bit
# # MAKE SURE TO CHANGE THIS DEPENDING ON WHERE I CHOOSE MY CONTROL POINTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i in range(18, len(obj) * 3):
    P_o[i][i] = 1 / sigma2_e

# __________________________________________________________________

# ***************** First Least Squares Iteration ******************

# # calculate Design Matrices **************************************
# A_e
A_e = A_e_matrix(nur, iop, eop, obj, indmat, constants)

# A_i
A_i = A_i_matrix(nur, iop, eop, obj, indmat, constants)

# A_o
A_o = A_o_matrix(nur, iop, eop, obj, indmat, constants)

# # calculate Misclosure *******************************************
# image misclosure
w = misclosure(nur, iop, eop, obj, indmat, pho_RH, constants)

# object misclosure, initial w_o is 0
w_o = [0 for i in range(0, int(nur[2]))]

# # calculate Normal Natrix ****************************************
# N_ee
N_ee = Normal(A_e, P_i, A_e)

# N_ei
N_ei = Normal(A_e, P_i, A_i)
N_ie = np.transpose(N_ei)

# N_eo, where N_oe = N_eo^T
N_eo = Normal(A_e, P_i, A_o)
N_oe = np.transpose(N_eo)

# N_ii
N_ii = Normal(A_i, P_i, A_i)

# N_io
N_io = Normal(A_i, P_i, A_o)
N_oi = np.transpose(N_io)

# N_oo
N_oo = Normal_oo(A_o, P_i, P_o)

# # create full Normal Matrix **************************************
# top row of N
N = N_ee.tolist()
N = np.concatenate((N, N_ei), axis=1)
N = np.concatenate((N, N_eo), axis=1)

# middle row of N
N_2 = N_ie.tolist()
N_2 = np.concatenate((N_2, N_ii), axis=1)
N_2 = np.concatenate((N_2, N_io), axis=1)

# last row of N
N_3 = N_oe.tolist()
N_3 = np.concatenate((N_3, N_oi), axis=1)
N_3 = np.concatenate((N_3, N_oo), axis=1)

# full N matrix
N = np.concatenate((N, N_2), axis=0)
N = np.concatenate((N, N_3), axis=0)

# # calculate u matrices *******************************************
# u_e
u_e = Normal_u(A_e, P_i, w)

# u_i
u_i = Normal_u(A_i, P_i, w)

# u_o
u_o = Normal_u_o(A_o, P_i, w, P_o, w_o)

# # create full u matrix *******************************************
u = u_e.tolist()
u = np.concatenate((u, u_i), axis=0)
u = np.concatenate((u, u_o), axis=0)
output(A_i, 'lab2_ini_A_i.csv')
output(A_e, 'lab2_ini_A_e.csv')
output(A_o, 'lab2_ini_A_o.csv')
output(N, 'lab2_ini_N.csv')
output(N_ee, 'lab2_ini_N_ee.csv')
output(N_ei, 'lab2_ini_N_ei.csv')
output(N_ie, 'lab2_ini_N_ie.csv')
output(N_eo, 'lab2_ini_N_eo.csv')
output(N_ii, 'lab2_ini_N_ii.csv')
output(N_io, 'lab2_ini_N_io.csv')
output(N_oi, 'lab2_ini_N_oi.csv')
output(N_oo, 'lab2_ini_N_oo.csv')
# # calculate Delta (edited by Osaze) ******************************
neg_N_inv = (-1 * np.linalg.pinv(N))
d = np.matmul(neg_N_inv, u)

# create a temp delta to make delta vals into degrees
tempd = []
for i in range(0, len(d)):
    tempd.append(d[i])

counter = 3

while (counter + 2) < (6 * len(eop)):
    tempd[counter] = np.degrees(d[counter])
    tempd[counter + 1] = np.degrees(d[counter + 1])
    tempd[counter + 2] = np.degrees(d[counter + 2])

    counter = counter + 6

d = tempd

# create empty x_0 (ie. initial vector of measurements to be corrected)
x_0 = []

# append ext points to x_0
for i in range(0, len(eop)):
    x_0.append(eop[i][2])
    x_0.append(eop[i][3])
    x_0.append(eop[i][4])
    x_0.append(eop[i][5])
    x_0.append(eop[i][6])
    x_0.append(eop[i][7])

# append iop points to x_0
for i in range(0, len(iop)):
    x_0.append(iop[i][1])
    x_0.append(iop[i][2])
    x_0.append(iop[i][3])

# append obj points to x_0
for i in range(0, len(obj)):
    x_0.append(obj[i][1])
    x_0.append(obj[i][2])
    x_0.append(obj[i][3])

# # outputting initial values
output(A_e, 'lab2_ini_A_e.csv')
output(A_i, 'lab2_ini_A_i.csv')
output(A_o, 'lab2_ini_A_o.csv')
output(w, 'lab2_ini_misclosure.csv')
output(P_i, 'lab2_ini_P_i.csv')
output(P_o, 'lab2_ini_P_o.csv')
output(N_ee, 'lab2_ini_N_ee.csv')
output(N_eo, 'lab2_ini_N_eo.csv')
output(N_oo, 'lab2_ini_N_oo.csv')
output(N, 'lab2_ini_N.csv')
output(u_e, 'lab2_ini_u_e.csv')
output(u_o, 'lab2_ini_u_o.csv')
output(u, 'lab2_ini_u.csv')
output(d, 'lab2_ini_delta.csv')
output(x_0, 'lab2_ini_x_0.csv')

# correct x_0 value with delta
xhat = np.array(x_0) + np.array(d)
# make x_0 into x_hat
x_0 = [i for i in xhat]

# __________________________________________________________________

# ******************** Least Squares Iterations ********************

# Set Criteria *****************************************************
# criteria is 0.001 mm, ie. 1 um, because the pixel spacing is in um
# each delta value should be less than criteria value
criteria = constants[4][0]
criteria_not_met = True

# Create Empty w and N Matrices for Iterations *********************
w_ite = []
N_ite = []

while_counter = 0

# iterate until all values of delta meet criteria
while criteria_not_met:

    # # input new values into temp matrices ************************
    # input new values of eop
    eop_ite = []
    t = len(eop)
    for i in range(0, len(eop)):
        # create temp vec too append to eop_ite
        eop_temp = []

        # ensure same imgID and camID as original file
        eop_temp.append(eop[i][0])
        eop_temp.append(eop[i][1])

        # add corrected eop values
        eop_temp.append(xhat[6 * i])
        eop_temp.append(xhat[6 * i + 1])
        eop_temp.append(xhat[6 * i + 2])
        eop_temp.append(xhat[6 * i + 3])
        eop_temp.append(xhat[6 * i + 4])
        eop_temp.append(xhat[6 * i + 5])

        eop_ite.append(eop_temp)

    # input new values of iop
    iop_ite = []
    t1 = len(iop)
    for i in range(0, len(iop)):
        # create temp vec to append to iop_ite
        iop_temp = []

        # append same camID as original file
        iop_temp.append(iop[i][0])

        # add corrected iop values
        iop_temp.append(xhat[len(eop) * 6 + 3 * i])
        iop_temp.append(xhat[len(eop) * 6 + 3 * i + 1])
        iop_temp.append(xhat[len(eop) * 6 + 3 * i + 2])

        iop_ite.append(iop_temp)

    # input new values of obj
    obj_ite = []
    t2 = len(obj)
    for i in range(0, len(obj)):
        # create temp vec to append to obj_temp
        obj_temp = []

        # ensure same ptID as original file
        obj_temp.append(obj[i][0])

        # add corrected obj values
        obj_temp.append(xhat[len(eop) * 6 + len(iop) * 3 + 3 * i])
        obj_temp.append(xhat[len(eop) * 6 + len(iop) * 3 + 3 * i + 1])
        obj_temp.append(xhat[len(eop) * 6 + len(iop) * 3 + 3 * i + 2])

        obj_ite.append(obj_temp)

    # # calculate Design Matrices **********************************
    # A_e (design matrix for EOPs)
    A_e_ite = A_e_matrix(nur, iop_ite, eop_ite, obj_ite, indmat, constants)

    # A_i (design matrix for IOPs)
    A_i_ite = A_i_matrix(nur, iop_ite, eop_ite, obj_ite, indmat, constants)

    # A_o (design matrix for object points)
    A_o_ite = A_o_matrix(nur, iop_ite, eop_ite, obj_ite, indmat, constants)

    # # calculate Misclosure ***************************************
    # image misclosure (to calculate u_e)
    w_ite = misclosure(nur, iop_ite, eop_ite, obj_ite, indmat, pho_RH, constants)

    # object misclosure (to calculate u_o)
    w_o_ite = []

    # create single column of obj points [X;Y;Z] to calculate w_o
    single_col_obj = []

    for i in range(0, len(obj)):
        single_col_obj.append(obj[i][1])
        single_col_obj.append(obj[i][2])
        single_col_obj.append(obj[i][3])

    for i in range(len(eop_ite) * 6, len(eop_ite) * 6 + len(single_col_obj)):
        w_o_ite.append(xhat[i] - single_col_obj[i - len(eop_ite) * 6])

    # # calculate Normal Natrix ************************************
    # N_ee
    N_ee_ite = Normal(A_e_ite, P_i, A_e_ite)

    # N_ei
    N_ei_ite = Normal(A_e_ite, P_i, A_i_ite)
    N_ie_ite = np.transpose(N_ei_ite)

    # N_eo, where N_oe = N_eo^T
    N_eo_ite = Normal(A_e_ite, P_i, A_o_ite)
    N_oe_ite = np.transpose(N_eo_ite)

    # N_ii
    N_ii_ite = Normal(A_i_ite, P_i, A_i_ite)

    # N_io
    N_io_ite = Normal(A_i_ite, P_i, A_o_ite)
    N_oi_ite = np.transpose(N_io_ite)

    # N_oo
    N_oo_ite = Normal_oo(A_o_ite, P_i, P_o)

    # # create full Normal Matrix **********************************
    # top row of N
    N_ite = N_ee_ite.tolist()
    N_ite = np.concatenate((N_ite, N_ei_ite), axis=1)
    N_ite = np.concatenate((N_ite, N_eo_ite), axis=1)

    # middle row of N
    N_2_ite = N_ie_ite.tolist()
    N_2_ite = np.concatenate((N_2_ite, N_ii_ite), axis=1)
    N_2_ite = np.concatenate((N_2_ite, N_io_ite), axis=1)

    # last row of N
    N_3_ite = N_oe_ite.tolist()
    N_3_ite = np.concatenate((N_3_ite, N_oi_ite), axis=1)
    N_3_ite = np.concatenate((N_3_ite, N_oo_ite), axis=1)

    # full N matrix
    N_ite = np.concatenate((N_ite, N_2_ite), axis=0)
    N_ite = np.concatenate((N_ite, N_3_ite), axis=0)

    # # calculate u matrices ***************************************
    # u_e
    u_e_ite = Normal_u(A_e_ite, P_i, w_ite)

    # u_i
    u_i_ite = Normal_u(A_i_ite, P_i, w_ite)

    # u_o
    u_o_ite = Normal_u_o(A_o_ite, P_i, w_ite, P_o, w_o_ite)

    # # create full u matrix ***************************************
    u_ite = u_e_ite.tolist()
    u_ite = np.concatenate((u_ite, u_i_ite), axis=0)
    u_ite = np.concatenate((u_ite, u_o_ite), axis=0)

    # # calculate Delta (edited by Osaze) **************************
    neg_N_inv_ite = (-1 * np.linalg.pinv(N_ite))
    d = np.matmul(neg_N_inv_ite, u_ite)

    tempd = []
    for i in range(0, len(d)):
        tempd.append(d[i])

    counter = 3

    while (counter + 2) < (6 * len(eop)):
        tempd[counter] = np.degrees(d[counter])
        tempd[counter + 1] = np.degrees(d[counter + 1])
        tempd[counter + 2] = np.degrees(d[counter + 2])

        counter = counter + 6

    d = tempd

    # # correct x_0 value with delta (edited by Osaze) *************
    xhat = np.array(x_0) + np.array(d)

    # # make x_0 into x_hat ****************************************
    x_0 = [i for i in xhat]

    # # check if criteria has been met  (edited by Osaze) **********
    counter = 0

    test = np.linalg.norm(d)
    if test > criteria:
        counter = counter + 1
    else:
        criteria_not_met = False

    # output second iteration
    while_counter = while_counter + 1
    print(while_counter)
    if while_counter == 3:
        output(A_e_ite, 'lab2_3_A_e.csv')
        output(A_i_ite, 'lab2_3_A_i.csv')
        output(A_o_ite, 'lab2_3_A_o3.csv')
        output(w, 'lab2_3_misclosure.csv')
        output(N_ee_ite, 'lab2_3_N_ee.csv')
        output(N_eo_ite, 'lab2_3_N_eo.csv')
        output(N_oo_ite, 'lab2_3_N_oo.csv')
        output(N_ite, 'lab2_3_N.csv')
        output(u_e_ite, 'lab2_3_u_e.csv')
        output(u_o_ite, 'lab2_3_u_o.csv')
        output(u_ite, 'lab2_3_u.csv')
        output(d, 'lab2_3_delta.csv')
        output(x_0, 'lab2_3_x_0.csv')

    if while_counter == 3:
        output(d, 'lab_2_3_delta.csv')


# ********************* End of ENGO 531 LAB 2 **********************
#
# # ******************** ENGO 531 LAB 1 ********************
#
# inte = csv2mat('lab1_int.int')
# con = csv2mat('lab1_con.con')
# exte = csv2mat('lab1_ext.ext')
# tie = csv2mat('lab1_tie.tie')
#
# # create single column of all tie and con values, [X, Y, Z, X, Y, Z, ...]
# contie = []
# for i in range(0, len(tie)):
#     contie.append(tie[i][1])
#     contie.append(tie[i][2])
#     contie.append(tie[i][3])
#
# for i in range(0, len(con)):
#     contie.append(con[i][1])
#     contie.append(con[i][2])
#     contie.append(con[i][3])
#
# # # constants format (ie. configuration data):
# # 1st row: image size in pixels (x, y)
# # 2nd row: pixel spacing in micrometers (x, y)
# # 3rd row: normal principal distance in mm and a '0' as a placeholder
# # 4th row: std for image point coordinates, and std for control point coordinates
# # 5th row: convergence criteria
# constants = csv2mat('constants.txt')
#
# # turn pho from LH to RH
# pho_RH = lh2rh(pho_LH, constants)
#
# # sort pho_RH by imageID
# pho_RH.sort(key=lambda x: x[1])
#
# # make large matrix with all necessary indices
# indmat = indexIDmat(pho_RH, con, tie, exte, inte)
#
# # calculate proper matrix sizes
# nur = calc_nur(pho_RH, exte, con, tie)
#
# # # *** FIRST LEAST SQUARES ITERATION *** # #
#
# # # calculate Design Matrices
# # A_e
# A_e = A_e_matrix(nur, inte, exte, con, tie, indmat)
#
# # A_o
# A_o = A_o_matrix(nur, inte, exte, con, tie, indmat)
#
# # misclosure
# w = misclosure(nur, inte, exte, con, tie, indmat, pho_RH, constants)
#
# # initial w_o is 0
# w_o = [0 for i in range(0, int(nur[2]))]
#
# # # prep for Normal matrix
# # standard deviation for control point coordinates
# sigma2_e = constants[3][1] ** 2
# P_e = weight(nur[0], sigma2_e)
#
# # standard deviation for image point coordinates, converted to mm
# sigma2_i = (constants[3][0] * (constants[1][0] / 1000)) ** 2
# P_i = weight(nur[0], sigma2_i)
#
# # sigma_GCP = 0.1mm, given in p26 of Bundle adjustment part 3 ANNOTATED 20210927
# sigma2_GCP = constants[3][1] ** 2
#
# # create matrix of 0's
# P_o = [[0 for i in range(0, nur[2])] for j in range(0, nur[2])]
#
# # apply weight matrix if control point/GCP, which will be for the bottom bit
# for i in range(len(tie) * 3, (len(tie) * 3) + (len(con) * 3)):
#     P_o[i][i] = 1 / sigma2_GCP
#
# # # calculating N of Normal matrices
# # N_ee
# N_ee = Normal_ee(A_e, P_i)
#
# # N_eo, where N_oe = N_eo^T
# N_eo = Normal_eo(A_e, A_o, P_i)
# N_oe = np.transpose(N_eo)
#
# # N_oo
# N_oo = Normal_oo(A_o, P_i, P_o)
#
# # # create normal equations matrix, first top left quarter of N is N_ee
# N = N_ee.tolist()
#
# # add second quarter, top right, of N, which is N_eo
# for i in range(0, len(N_eo)):
#     for j in range(0, len(N_eo[0])):
#         N_app = N_eo[i][j].tolist()
#         N[i].append(N_app)
#
# # create bottom left quarter, which is N_oe
# N_bottom = N_oe.tolist()
#
# # add bottom right quarter
# for i in range(0, len(N_oo)):
#     for j in range(0, len(N_oo[0])):
#         N_app = N_oo[i][j].tolist()
#         N_bottom[i].append(N_app)
#
# # append top N and bottom N together
# [N.append(N_bottom[i]) for i in range(0, len(N_bottom))]
#
# # # calculating u of Normal matrices
# # u_e
# u_e = Normal_u_e(A_e, P_i, w)
#
# # u_o
# u_o = Normal_u_o(A_o, P_i, w, P_o, w_o)
#
# # create u, top half
# u = u_e.tolist()
#
# # append top u, u_e, and bottom u, u_o
# [u.append(u_o[i]) for i in range(0, len(u_o))]
#
# # calculate delta
#
# # Edited by Osaze
# neg_N_inv = (-1 * np.linalg.inv(N))
# output(N, 'checkingN.csv')
# d = np.matmul(neg_N_inv, u)
#
# tempd = []
# for i in range(0, len(d)):
#     tempd.append(d[i])
#
# counter = 3
#
# while (counter + 2) < (6 * len(exte)):
#     tempd[counter] = np.degrees(d[counter])
#     tempd[counter + 1] = np.degrees(d[counter + 1])
#     tempd[counter + 2] = np.degrees(d[counter + 2])
#
#     counter = counter + 6
#
# d = tempd
# output(np.linalg.inv(N), 'ini_N_inverse.csv')
#
# # Edited by Osaze
# # create empty x_0 matrix (ie. initial vector of measurements to be corrected)
# x_0 = []
#
# # append ext points to x_0
# for i in range(0, len(exte)):
#     x_0.append(exte[i][2])
#     x_0.append(exte[i][3])
#     x_0.append(exte[i][4])
#     x_0.append(exte[i][5])
#     x_0.append(exte[i][6])
#     x_0.append(exte[i][7])
#
# # append tie points to x_0
# for i in range(0, len(tie)):
#     x_0.append(tie[i][1])
#     x_0.append(tie[i][2])
#     x_0.append(tie[i][3])
#
# # append con points to x_0
# for i in range(0, len(con)):
#     x_0.append(con[i][1])
#     x_0.append(con[i][2])
#     x_0.append(con[i][3])
#
# # # outputting initial values
# output(A_e, 'ini_A_e.csv')
# output(A_o, 'ini_A_o.csv')
# output(w, 'ini_misclosure.csv')
# output(P_i, 'ini_P_i.csv')
# output(P_o, 'ini_P_o.csv')
# output(N_ee, 'ini_N_ee.csv')
# output(N_eo, 'ini_N_eo.csv')
# output(N_oo, 'ini_N_oo.csv')
# output(N, 'ini_N.csv')
# output(u_e, 'ini_u_e.csv')
# output(u_o, 'ini_u_o.csv')
# output(u, 'ini_u.csv')
# output(d, 'ini_delta.csv')
# output(x_0, 'ini_x_0.csv')
#
# # correct x_0 value with delta
# xhat = np.array(x_0) + np.array(d)
# # make x_0 into x_hat
# x_0 = [i for i in xhat]
#
# # criteria is 0.001 mm, ie. 1 um, because the pixel spacing is in um
# # each delta value should be less than criteria value
# criteria = constants[4][0]
# criteria_not_met = True
#
# w_ite = []
# N_ite = []
#
# while_counter = 0
#
# # iterate until all values of delta meet criteria
# while criteria_not_met:
#
#     # # input new values into exte
#     exte_ite = []
#
#     for i in range(0, len(exte)):
#         # create temp vec too append to exte_ite
#         exte_temp = []
#
#         # ensure same imageID and cameraID as original file
#         exte_temp.append(exte[i][0])
#         exte_temp.append(exte[i][1])
#
#         # add corrected exte values
#         exte_temp.append(xhat[6 * i])
#         exte_temp.append(xhat[6 * i + 1])
#         exte_temp.append(xhat[6 * i + 2])
#         exte_temp.append(xhat[6 * i + 3])
#         exte_temp.append(xhat[6 * i + 4])
#         exte_temp.append(xhat[6 * i + 5])
#
#         exte_ite.append(exte_temp)
#
#     # # input new values into tie
#     tie_ite = []
#
#     for i in range(0, len(tie)):
#         # create temp vec too append to tie_ite
#         tie_temp = []
#
#         # ensure same pointID as original file
#         tie_temp.append(tie[i][0])
#
#         # add corrected tie values
#         tie_temp.append(xhat[len(exte) * 6 + 3 * i])
#         tie_temp.append(xhat[len(exte) * 6 + 3 * i + 1])
#         tie_temp.append(xhat[len(exte) * 6 + 3 * i + 2])
#
#         tie_ite.append(tie_temp)
#
#     # # input new values into con
#     con_ite = []
#
#     for i in range(0, len(con)):
#         # create temp vec too append to con_tie
#         con_temp = []
#
#         # ensure same pointID as original file
#         con_temp.append(con[i][0])
#
#         # add corrected exte values
#         con_temp.append(xhat[len(exte) * 6 + len(tie) * 3 + 3 * i])
#         con_temp.append(xhat[len(exte) * 6 + len(tie) * 3 + 3 * i + 1])
#         con_temp.append(xhat[len(exte) * 6 + len(tie) * 3 + 3 * i + 2])
#
#         con_ite.append(con_temp)
#
#     # create design matrix for EOP's
#     A_e_ite = A_e_matrix(nur, inte, exte_ite, con_ite, tie_ite, indmat)
#
#     # create design matrix for object points
#     A_o_ite = A_o_matrix(nur, inte, exte_ite, con_ite, tie_ite, indmat)
#
#     # misclosure (to calculate u_e)
#     w_ite = misclosure(nur, inte, exte_ite, con_ite, tie_ite, indmat, pho_RH, constants)
#
#     # misclosure (to calculate u_o)
#     w_o_ite = []
#     for i in range(len(exte_ite) * 6, len(exte_ite) * 6 + len(contie)):
#         w_o_ite.append(xhat[i] - contie[i - len(exte_ite) * 6])
#
#     # # calculating N of Normal matrices
#     # N_ee
#     N_ee_ite = Normal_ee(A_e_ite, P_i)
#
#     # N_eo, where N_oe = N_eo^T
#     N_eo_ite = Normal_eo(A_e_ite, A_o_ite, P_i)
#     N_oe_ite = np.transpose(N_eo_ite)
#
#     # N_oo
#     N_oo_ite = Normal_oo(A_o_ite, P_i, P_o)
#
#     # # create normal equations matrix, first top left quarter of N is N_ee
#     N_ite = N_ee_ite.tolist()
#
#     # add second quarter, top right, of N, which is N_eo
#     for i in range(0, len(N_eo_ite)):
#         for j in range(0, len(N_eo_ite[0])):
#             N_app_ite = N_eo_ite[i][j].tolist()
#             N_ite[i].append(N_app_ite)
#
#     # create bottom left quarter, which is N_oe
#     N_bottom_ite = N_oe_ite.tolist()
#
#     # add bottom right quarter
#     for i in range(0, len(N_oo_ite)):
#         for j in range(0, len(N_oo_ite[0])):
#             N_app_ite = N_oo_ite[i][j].tolist()
#             N_bottom_ite[i].append(N_app_ite)
#
#     # append top N and bottom N together
#     [N_ite.append(N_bottom_ite[i]) for i in range(0, len(N_bottom_ite))]
#
#     # # calculating u of Normal matrices
#
#     # u_e
#     u_e_ite = Normal_u_e(A_e_ite, P_i, w_ite)
#
#     # u_o
#     u_o_ite = Normal_u_o(A_o_ite, P_i, w_ite, P_o, w_o_ite)
#
#     # create u, top half
#     u_ite = u_e_ite.tolist()
#
#     # append top u, u_e, and bottom u, u_o
#     [u_ite.append(u_o_ite[i]) for i in range(0, len(u_o_ite))]
#
#     # Edited by Osaze
#     # calculate delta
#     d = -np.matmul(np.linalg.inv(N_ite), u_ite)
#
#     tempd = []
#     for i in range(0, len(d)):
#         tempd.append(d[i])
#
#     counter = 3
#
#     while (counter + 2) < (6 * len(exte)):
#         tempd[counter] = np.degrees(d[counter])
#         tempd[counter + 1] = np.degrees(d[counter + 1])
#         tempd[counter + 2] = np.degrees(d[counter + 2])
#
#         counter = counter + 6
#
#     d = tempd
#
#     # Edited by Osaze
#     # correct x_0 value with delta
#     xhat = np.array(x_0) + np.array(d)
#
#     # make x_0 into x_hat
#     x_0 = [i for i in xhat]
#
#     # check if criteria has been met
#     # Edited by Osaze
#     counter = 0
#     # for i in range(0, len(d)):
#     test = np.linalg.norm(d)
#     if test > criteria:
#         counter = counter + 1
#     else:
#         criteria_not_met = False
#         # if counter == len(d):
#         # criteria_not_met = False
#     while_counter = while_counter + 1
#     print(while_counter)
#     if while_counter == 1:
#         output(A_e, '1_A_e.csv')
#         output(A_o, '1_A_o.csv')
#         output(w, '1_misclosure.csv')
#         output(P_i, '1_P_i.csv')
#         output(P_o, '1_P_o.csv')
#         output(N_ee, '1_N_ee.csv')
#         output(N_eo, '1_N_eo.csv')
#         output(N_oo, '1_N_oo.csv')
#         output(N, '1_N.csv')
#         output(u_e, '1_u_e.csv')
#         output(u_o, '1_u_o.csv')
#         output(u, '1_u.csv')
#         output(d, '1_delta.csv')
#         output(x_0, '1_x_0.csv')
#         output(exte_ite, '1_exte_ite.csv')
#         output(tie_ite, '1_tie_ite.csv')
#         output(con_ite, '1_con_ite.csv')
#
#     if while_counter == 3:
#         output(d, '3_delta.csv')
#
# # output exte
# exte_output = []
# for i in range(0, len(exte) * 6, 6):
#     ind = int(i / 6)
#     exte_output.append([exte[ind][0], exte[ind][1],
#                         xhat[i], xhat[i + 1], xhat[i + 2], xhat[i + 3], xhat[i + 4], xhat[i + 5]])
#
# # outpute tie
# tie_output = []
# for i in range(len(exte) * 6, len(exte) * 6 + len(tie) * 3, 3):
#     ind = int((i - len(exte) * 6) / 3)
#     tie_output.append([tie[ind][0], xhat[i], xhat[i + 1], xhat[i + 2]])
#
# # output con
# con_output = []
# for i in range(len(exte) * 6 + len(tie) * 3, len(exte) * 6 + len(tie) * 3 + len(con) * 3, 3):
#     ind = int((i - (len(exte) * 6 + len(tie) * 3)) / 3)
#     con_output.append([con[ind][0], xhat[i], xhat[i + 1], xhat[i + 2]])
#
# # # *** VERIFICATION *** # #
#
# # turn A_e and A_o into A
# A = []
# for i in range(0, len(A_e_ite)):
#     A_temp = []
#     for j in range(0, len(A_e_ite[0])):
#         A_temp.append(A_e_ite[i][j])
#
#     for j in range(0, len(A_o[0])):
#         A_temp.append(A_o_ite[i][j])
#
#     A.append(A_temp)
#
# # calculate the residual
# # vhat = A * delta + w
# vhat = np.matmul(A, d)
# vhat = vhat + w_ite
#
# # estimated variance factor
# sigma2_0 = np.matmul(np.transpose(vhat), P_i)
# sigma2_0 = np.matmul(sigma2_0, vhat)
# sigma2_0 = [sigma2_0 / (nur[0] - (nur[1] + nur[2]))]
#
# # covariance matrix of the estimated parameters
# Cxhat = np.linalg.inv(N_ite)
#
# # covariance matrix of the estimated residuals
# Cl = sigma2_i * np.linalg.inv(P_i)
#
# AT = np.transpose(A)
# Cvhat = np.matmul(AT, P_i)
# Cvhat = np.matmul(Cvhat, A)
# Cvhat = np.linalg.inv(Cvhat)
# Cvhat = np.matmul(A, Cvhat)
# Cvhat = np.matmul(Cvhat, AT)
# Cvhat = Cl - Cvhat
#
# # final output of files
# output(A, 'A Final.csv')
# output(sigma2_0, 'sigma2_0.csv')
# output(vhat, 'vhat.csv')
# output(Cxhat, 'Cxhat.csv')
# output(Cvhat, 'Cvhat.csv')
#
# final_output_file(pho_RH, inte, con, exte, tie, vhat, sigma2_0, xhat)
