import numpy as np
import pandas as pd
import csv
import math


def csv2mat(filename):
    # read CSV file and store as matrix

    # open file
    csvfile = open(filename)

    # read csvfile into DataFrame
    mat_df = pd.read_csv(csvfile, header=None, delim_whitespace=True)

    # turn DataFrame to list
    mat = mat_df.values.tolist()

    # create matrix to be returned
    ret_mat = []

    # remove all empty rows and excess tabs/spaces
    for i in range(0, len(mat)):

        # only deal with row if it's not empty
        if mat[i]:

            # create temporary row
            tempmat = []

            for j in range(0, len(mat[i])):

                # check if there is an empty space
                if mat[i][j] != '':
                    tempmat.append(mat[i][j])

            # append temp mat to mat that will be returned
            ret_mat.append(tempmat)

    # close file
    csvfile.close()

    return ret_mat


def output(mat, output_filename):
    # output mat to file called filename
    if len(mat) > 1:
        np.savetxt(output_filename, mat, delimiter=", ", fmt='%s')
    else:
        np.savetxt(output_filename, mat)


def take_col(mat, ind):
    # return a list that has only one of the columns from mat in the index ind

    # create empty list for column elements
    col_list = []

    for i in range(0, len(mat)):
        col_list.append(mat[i][ind])

    return col_list


def lh2rh(LH_coord, constants):
    # (Assume it's always .pho that's passed) LH_coord has: imageID; cameraID; x;y

    # constants format:
    # 1st row: image size in pixels (x, y)
    # 2nd row: pixel spacing in micrometers (x, y)
    # 3rd row: normal principal distance in mm and a '0' as a placeholder

    # create empty list for valueS in RH coordinate system
    RH_coord = []

    for i in range(0, len(LH_coord)):
        # calc x coordinate in RH: x_ij = [n_ij - (N/2 - 0.5)] * delta_x
        # *** USING -0.5 ***
        x_ij = (LH_coord[i][2] - (constants[0][0] / 2 - 0.5)) * constants[1][0] / 1000

        # calc y coordinate in RH: y_ij = [n_ij - (N/2 - 0.5)] * delta_y
        # *** USING -0.5 ***
        y_ij = ((constants[0][1] / 2 - 0.5) - LH_coord[i][3]) * constants[1][1] / 1000

        # append all values to RH_coord, including image and point IDs
        RH_coord.append([LH_coord[i][0], LH_coord[i][1], x_ij, y_ij])

    return RH_coord


def id_index(mat_col, IDnum):
    # returns index of desired ID number
    # mat_col is a list with all relevant ID's
    # IDnum is the ID number desired

    index = None

    # checks if IDnum is a float; if it is, turn it into an int
    if isinstance(IDnum, float):
        IDnum = int(IDnum)

    for i in range(0, len(mat_col)):

        # check if ID in mat_col is a float, then turn it into an int
        # this is for cameraID's (eg. str(99.0) != str(99))
        if isinstance(mat_col[i], float):
            mat_col[i] = int(mat_col[i])

        if str(mat_col[i]) == str(IDnum):
            # change index value to row # of matching ID number
            index = i
            break

    return index


def indexIDmat(phomat, objmat, extmat, intmat):
    # this functions finds the index of the imageID in extmat that's correlated to the phomat point
    # then find the index of the camID in intmat that's correlated to the iopmat image thru the phomat point

    # create matrix that will have all relevant indexes and info
    # will have: pointID; obj column (replace tie/con); pointID index; imageID index (extmat); camID index (intmat)

    # create empty matrix
    indmat = []

    # go through each row in pho
    for i in range(0, len(phomat)):

        indmat_temp = []

        # match pointID to point in objmat, index = 0
        obj_ind = id_index(take_col(objmat, 0), phomat[i][0])

        # label in indmat_temp that the point is a obj and the index of the matching obj point
        indmat_temp.append('obj')
        indmat_temp.append(obj_ind)

        # find matching imageID in extmat
        imID_ind = id_index(take_col(extmat, 0), phomat[i][1])
        indmat_temp.append(imID_ind)

        # find matching cameraID in intmat
        t = take_col(intmat, 0)
        test = extmat[imID_ind][1]
        caID_ind = id_index(take_col(intmat, 0), extmat[imID_ind][1])
        indmat_temp.append(caID_ind)

        # append indmat vector to final indmat matrix
        indmat.append(indmat_temp)

    # indmat should not have headers but the output of indmat should
    # therefore, indmat will be copied to another variable which will have headers and then outputted
    indmat_output = [['Point Type', 'Index (Point ID)', 'Index (Image ID)', 'Index (Camera ID)']]
    indmat_output.extend(indmat)

    # output indmat_output to .csv file
    output(indmat_output, 'Index of all files LAB 2.csv')

    return indmat


def calc_nur(phomat, extmat, objmat):
    # # used to calculate dimension of design matrix
    # dim(A_e) = n_p x u_e
    # dim(A_o) = n_p x u_o

    # n_p is the number of image point coordinate observations
    n_p = 2 * len(phomat)

    # u_e is the number of EOP
    u_e = 6 * len(extmat)

    # u_o is the number of object point unknowns
    u_o = 3 * len(objmat)

    # inputting values into matrix that will be returned
    nur = [n_p, u_e, u_o]

    # return format will be: n_p, u_e, and u_o
    return nur


def rot_mat(extmat_row):
    # rotation matrix calculated using formula on Bundle adjustment part 1 slides on D2L
    # extmat must be in deg
    # M_j = R_3(kap)*R_2(phi)*R_1(omg)

    # create empty matrix
    rotmat = []

    # use new variables to store angle values
    omg = extmat_row[5]
    phi = extmat_row[6]
    kap = extmat_row[7]

    # make omg, phi, and kap into rad
    omg = math.radians(omg)
    phi = math.radians(phi)
    kap = math.radians(kap)

    # 1st row of rotation matrix, will have 3 columns
    rotmat.append([math.cos(phi) * math.cos(kap),
                   math.cos(omg) * math.sin(kap) + math.sin(omg) * math.sin(phi) * math.cos(kap),
                   math.sin(omg) * math.sin(kap) - math.cos(omg) * math.sin(phi) * math.cos(kap)])

    # 2nd row of rotation matrix, will have 3 columns
    rotmat.append([-math.cos(phi) * math.sin(kap),
                   math.cos(omg) * math.cos(kap) - math.sin(omg) * math.sin(phi) * math.sin(kap),
                   math.sin(omg) * math.cos(kap) + math.cos(omg) * math.sin(phi) * math.sin(kap)])

    # 3rd row of rotation matrix, will have 3 columns
    rotmat.append([math.sin(phi),
                   -math.sin(omg) * math.cos(phi),
                   math.cos(omg) * math.cos(phi)])

    # check that rotation matrix is correct
    if abs(math.atan(-rotmat[2][1] / rotmat[2][2])) - abs(omg) > 0.00001:
        print(math.atan(-rotmat[2][1] / rotmat[2][2]) - omg)
        print("calc_omg - omg: ", math.atan(-rotmat[2][1] / rotmat[2][2]), " - ", omg)
        print("rotmat is incorrect when checking omg")
    if abs(math.asin(rotmat[2][0])) - abs(phi) > 0.00001:
        print(math.asin(rotmat[2][0]) - phi)
        print("calc_phi - phi: ", math.asin(rotmat[2][0]), " - ", phi)
        print("rotmat is incorrect when checking phi")
    if abs(math.atan(-rotmat[1][0] / rotmat[0][0])) - abs(kap) > 0.00001:
        print(math.atan(-rotmat[1][0] / rotmat[0][0]) - kap)
        print("calc_kap - kap: ", math.atan(-rotmat[1][0] / rotmat[0][0]), " - ", kap)
        print("rotmat is incorrect when checking kap")

    return rotmat


def A_mat_matchval(indmat, i, objmat, extmat, intmat, constants):
    # take pointID from indmat and find corresponding XYZ**C value from extmat and XYZ values from

    # find index of objmat using indmat
    X = objmat[indmat[i][1]][1]
    Y = objmat[indmat[i][1]][2]
    Z = objmat[indmat[i][1]][3]

    # finding points for XYZ**C using imageID
    Xc = extmat[indmat[i][2]][2]
    Yc = extmat[indmat[i][2]][3]
    Zc = extmat[indmat[i][2]][4]
    t=indmat[i][3]
    # assign principal distance from intmat
    c = intmat[indmat[i][3]][3]

    # turn c from pixels to mm
    c = c * constants[1][0] / 1000

    # find omega, phi, and kappa from extmat and convert to rad
    omg = extmat[indmat[i][2]][5]
    phi = extmat[indmat[i][2]][6]
    kap = extmat[indmat[i][2]][7]

    # add values to return matrix
    vals = [X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap]

    return vals


def A_mat_calcs(M, X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap):
    # calculate dx/dXc, dx/dYc, dx/dZc, dy/dXc, dy/dYc, and dy/dZc
    # aka -dx/dX, -dx/dY, -dx/dZ, -dy/dX, -dy/dY, and -dy/dZ respectively
    # also calculates dx/domg, dx/dphi, dx/dkap, dy/domg, dy/dphi, and dy/dkap
    # omg, phi, and kap should be in rad

    # temporarily fill A_e x and y rows with 0's
    A = [[None, None, None, None, None, None], [None, None, None, None, None, None]]

    # calculate UVW
    U = M[0][0] * (X - Xc) + M[0][1] * (Y - Yc) + M[0][2] * (Z - Zc)
    V = M[1][0] * (X - Xc) + M[1][1] * (Y - Yc) + M[1][2] * (Z - Zc)
    W = M[2][0] * (X - Xc) + M[2][1] * (Y - Yc) + M[2][2] * (Z - Zc)

    somg = math.sin(omg)
    sphi = math.sin(phi)
    skap = math.sin(kap)

    comg = math.cos(omg)
    cphi = math.cos(phi)
    ckap = math.cos(kap)

    # find dx/d_ values
    dx_dphi1 = (X - Xc) * (-W * sphi * ckap - U * cphi)
    dx_dphi2 = (Y - Yc) * (W * somg * cphi * ckap - U * somg * sphi)
    dx_dphi3 = (Z - Zc) * (-W * comg * cphi * ckap + U * comg * sphi)

    A[0] = [-c / (W ** 2) * (M[2][0] * U - M[0][0] * W),
            -c / (W ** 2) * (M[2][1] * U - M[0][1] * W),
            -c / (W ** 2) * (M[2][2] * U - M[0][2] * W),
            -c / (W ** 2) * ((Y - Yc) * (U * M[2][2] - W * M[0][2]) - (Z - Zc) * (U * M[2][1] - W * M[0][1])),
            -c / (W ** 2) * (dx_dphi1 + dx_dphi2 + dx_dphi3),
            - c * V / W]

    # find dy/d_ values
    dy_dphi1 = (X - Xc) * (W * sphi * skap - V * cphi)
    dy_dphi2 = (Y - Yc) * (-W * somg * cphi * skap - V * somg * sphi)
    dy_dphi3 = (Z - Zc) * (W * comg * cphi * skap + V * comg * sphi)

    A[1] = [-c / (W ** 2) * (M[2][0] * V - M[1][0] * W),
            -c / (W ** 2) * (M[2][1] * V - M[1][1] * W),
            -c / (W ** 2) * (M[2][2] * V - M[1][2] * W),
            -c / (W ** 2) * ((Y - Yc) * (V * M[2][2] - W * M[1][2])
                             - (Z - Zc) * (V * M[2][1] - W * M[1][1])),
            -c / (W ** 2) * (dy_dphi1 + dy_dphi2 + dy_dphi3),
            c * U / W]

    return A


def find_UVW(M, X, Y, Z, Xc, Yc, Zc):
    # calculate UVW
    U = M[0][0] * (X - Xc) + M[0][1] * (Y - Yc) + M[0][2] * (Z - Zc)
    V = M[1][0] * (X - Xc) + M[1][1] * (Y - Yc) + M[1][2] * (Z - Zc)
    W = M[2][0] * (X - Xc) + M[2][1] * (Y - Yc) + M[2][2] * (Z - Zc)

    UVW = [U, V, W]

    return UVW


def A_e_matrix(nur, intmat, extmat, objmat, indmat, constants):
    # calculate exterior design matrix
    # reminder that indmat is formatted like: pointType; Ind(ptID); Ind(imID); Ind(caID)

    # create empty matrix for A_e
    A_e = []

    # go through each observation/row in A_e
    for i in range(0, int(nur[0] / 2)):

        # create empty temp row to append to A_e
        A_e_rowX = []
        A_e_rowY = []

        # find rotation matrix of row i
        M = rot_mat(extmat[indmat[i][2]])

        # get values
        [X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap] = A_mat_matchval(indmat, i, objmat, extmat, intmat, constants)

        omg_c = math.radians(omg)
        phi_c = math.radians(phi)
        kap_c = math.radians(kap)

        # go through each column
        # it's divided by 6 bc each section will be filled by 6 elements every time
        for j in range(0, int(nur[1] / 6)):

            # the imageID from extmat goes to the next imageID
            imID = extmat[j][0]

            # if imageID does not match parameter ID, that element will be 0
            if extmat[indmat[i][2]][0] != imID:
                A_e_rowX.extend([0, 0, 0, 0, 0, 0])
                A_e_rowY.extend([0, 0, 0, 0, 0, 0])

            # if imageID does match parameter ID, calculate values
            else:
                A_e_xy = A_mat_calcs(M, X, Y, Z, Xc, Yc, Zc, c, omg_c, phi_c, kap_c)
                A_e_rowX.extend(A_e_xy[0])
                A_e_rowY.extend(A_e_xy[1])

        A_e.append(A_e_rowX)
        A_e.append(A_e_rowY)

    return A_e


def A_o_matrix(nur, intmat, extmat, objmat, indmat, constants):
    # calculate object design matrix
    # size of A_o is (# of image point observations in phomat * 2) x (# of obj points * 3)
    # reminder that indmat is formatted like: pointType; Ind(ptID); Ind(imID); Ind(caID)

    # create empty A_o matrix
    A_o = []
    #contie = csv2mat('lab1_contie.txt')

    # go thru for loop every 2 rows bc x row and y row will be calculated at the same time
    for i in range(0, int(nur[0] / 2)):

        # create empty temp row to append to A_e
        A_o_rowX = []
        A_o_rowY = []

        # find rotation matrix of row i
        M = rot_mat(extmat[indmat[i][2]])

        # get values
        [X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap] = A_mat_matchval(indmat, i, objmat, extmat, intmat, constants)

        omg = math.radians(omg)
        phi = math.radians(phi)
        kap = math.radians(kap)

        # go thru for loop every 3 columns X Y and Z columns will be calculated at same time
        for j in range(0, int(nur[2] / 3)):
            ptID = objmat[indmat[i][1]][0]

            if objmat[j][0] != ptID:
                A_o_rowX.extend([0, 0, 0])
                A_o_rowY.extend([0, 0, 0])
            else:
                A_e_xy = A_mat_calcs(M, X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap)
                A_o_rowX.extend([-A_e_xy[0][0], -A_e_xy[0][1], -A_e_xy[0][2]])
                A_o_rowY.extend([-A_e_xy[1][0], -A_e_xy[1][1], -A_e_xy[1][2]])

        A_o.append(A_o_rowX)
        A_o.append(A_o_rowY)

    return A_o


def A_i_matrix(nur, intmat, extmat, objmat, indmat, constants):
    # calculate image design matrix
    # size of A_i is (# of image point observations in pho * 2 [x and y]) x (# of obj points [given on D2L] * 3 [xyz])

    # create empty A_i matrix
    A_i = []

    # go thru for loop every 2 rows bc x row and y row will be calculated at the same time
    for i in range(0, int(nur[0] / 2)):
        A_i_rowX = []
        A_i_rowY = []

        # find rotation matrix of rpw i
        M = rot_mat(extmat[indmat[i][2]])

        # get values
        [X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap] = A_mat_matchval(indmat, i, objmat, extmat, intmat, constants)

        [U, V, W] = find_UVW(M, X, Y, Z, Xc, Yc, Zc)

        A_i.append([1, 0, -U / W])
        A_i.append([0, 1, -V / W])

    return A_i


def calculated(nur, intmat, extmat, objmat, indmat, constants):
    # calculate the misclosure
    x0 = []

    for i in range(0, int(nur[0] / 2)):
        [X, Y, Z, Xc, Yc, Zc, c, omg, phi, kap] = A_mat_matchval(indmat, i, objmat, extmat, intmat, constants)

        extmat_row = [0, 0, 0, 0, 0, omg, phi, kap]

        M = rot_mat(extmat_row)

        U = M[0][0] * (X - Xc) + M[0][1] * (Y - Yc) + M[0][2] * (Z - Zc)
        V = M[1][0] * (X - Xc) + M[1][1] * (Y - Yc) + M[1][2] * (Z - Zc)
        W = M[2][0] * (X - Xc) + M[2][1] * (Y - Yc) + M[2][2] * (Z - Zc)

        # calculate RH of xp and yp
        xp_yp = [[0, 0, intmat[indmat[i][3]][1], intmat[indmat[i][3]][2]]]

        # turn into right hand coordinates
        RH = lh2rh(xp_yp, constants)
        x_p = RH[0][2]
        y_p = RH[0][3]

        # append x_ij
        x0.append(x_p - c * U / W)
        # append y_ij
        x0.append(y_p - c * V / W)

    return x0


def misclosure(nur, intmat, extmat, objmat, indmat, phomat, constants):
    # find calculated x_ij and y_ij

    # find calculated x_ij and y_ij values
    x0 = calculated(nur, intmat, extmat, objmat, indmat, constants)

    # check that matrix size is right
    if len(x0) != len(phomat) * 2:
        print("Matrix Length of Calculated Values != 2 * Matrix Length of .pho")

    # create empty return matrix
    w = []

    # misclosure calculation
    for i in range(0, len(x0), 2):
        if i == 0:
            w.append(x0[i] - phomat[i][2])
            w.append(x0[i + 1] - phomat[i][3])
        else:
            pho_ind = int(i / 2)
            w.append(x0[i] - phomat[pho_ind][2])
            w.append(x0[i + 1] - phomat[pho_ind][3])
    return w


def weight(mat_size, sigma2):
    # weight matrix, P = 1/Cl = 1/sigma^2

    # create a 0 matrix
    P = [[0 for _ in range(0, mat_size)] for _ in range(0, mat_size)]

    # make all diagonal elements into sigma^2
    for i in range(0, mat_size):
        P[i][i] = 1 / sigma2

    return P


def Normal(A, P, A_2):
    # calculate Normal matrix (generic)
    # N = A^T * P * A_2

    # calculate N
    N = np.transpose(A)

    N = np.matmul(N, P)
    N = np.matmul(N, A_2)

    return N


def Normal_u(A, P, w):
    # finding u_e in system of Normal equations
    # u_e = A_e^T * P * w

    # calculate u_e
    u = np.transpose(A)
    u = np.matmul(u, P)
    u = np.matmul(u, w)

    return u


def Normal_oo(A_o, P_i, P_o):
    # calculate Object Point Normal matrix in system of Normal equations
    # N_oo - A_o^T * P * A_o + P_o

    # calculate N_oo
    N_oo = np.transpose(A_o)
    N_oo = np.matmul(N_oo, P_i)
    N_oo = np.matmul(N_oo, A_o)
    N_oo = N_oo + P_o

    return N_oo


def Normal_u_o(A_o, P_i, w, P_o, w_o):
    # finding u_o in system of Normal equations
    # weighted
    # u_o = A_o^T * P * w + P_o * w_o

    # calculate u_o
    u_o = np.transpose(A_o)
    u_o = np.matmul(u_o, P_i)
    u_o = np.matmul(u_o, w)

    u_o = u_o + np.matmul(P_o, w_o)

    return u_o


def final_output_file(phomat, intmat, extmat, objmat, vhat, sigma2_0, xhat):
    output_file = [["Apekhchya Shrestha - ENGO 531 Lab 1:  Bundle Adjustment Software Construction"],
                   ["Course instructor: Dr. Derek Lichti"],
                   ["Teaching Assistant: Sandra Simenova"], [],
                   ["******************************************"],
                   ["EXECUTION DATE: October 12 2021"],
                   ["******************************************"], [],
                   ["Number of EOP's:\t\t\t\t", len(extmat)], ["Number of EOP unknowns:\t\t\t", len(extmat) * 6],
                   ["Number of cameras used:\t\t\t", len(intmat)],
                   ["Number of IOP unknowns:\t\t\t", len(intmat) * 0], ["Number of obj points:\t\t\t\t", len(objmat)],
                   ["Number of obj point unknowns:\t\t\t", len(objmat) * 3],
                   ["-------------------------------------------"],
                   (["Total Number of Unknowns:\t\t\t", len(extmat) * 6 + len(objmat) * 3]), [],
                   ["******************************************"], [],
                   ["Number of observed image points:\t", len(phomat)],
                   ["Number of image point coord obs:\t", len(phomat) * 2], ["Number of EOP observations:\t\t", 0],
                   ["-------------------------------------------"],
                   ["Total Number of Observations:\t\t", len(phomat) * 2], [],
                   ["******************************************"], [],
                   ["Degrees of freedom:\t\t\t\t\t", (len(phomat) * 2)
                    - (len(extmat) * 6 + len(objmat) * 3)], [],
                   ["******************************************"], [], ["Residuals"],
                   ["-------------------------------------------"], ["PointID\tImageID\t\t\tvx\t\t\tvy"],
                   ["-------------------------------------------"]]

    for i in range(0, len(vhat), 2):
        if i == 0:
            output_file.append([phomat[i][0], "\t", phomat[i][1], "\t", vhat[i], "\t", vhat[i + 1]])
        else:
            output_file.append([phomat[int(i / 2)][0], "\t", phomat[int(i / 2)][1], "\t", vhat[i], "\t", vhat[i + 1]])
    output_file.append(["-------------------------------------------"])

    # calculate RMS of residuals
    RMS_x = 0
    res_mean_x = 0
    for i in range(0, len(vhat), 2):
        RMS_x = RMS_x + vhat[i] ** 2
        res_mean_x = res_mean_x + vhat[i]
    RMS_x = (1 / (len(vhat) / 2) * RMS_x) ** (1 / 2)

    RMS_y = 0
    res_mean_y = 0
    for i in range(1, len(vhat), 2):
        RMS_y = RMS_y + vhat[i] ** 2
        res_mean_y = res_mean_y + vhat[i]
    RMS_y = (1 / (len(vhat) / 2) * RMS_y) ** (1 / 2)

    output_file.append(["RMS =\t\t\t", RMS_x, "\t", RMS_y])
    output_file.append(["Mean of Residuals = \t", res_mean_x, "\t", res_mean_y])
    output_file.append(["-------------------------------------------"])

    output_file.append([])
    output_file.append(["******************************************"])
    output_file.append([])

    output_file.append(["Estimated variance factor = ", sigma2_0[0]])

    output_file.append([])
    output_file.append(["******************************************"])
    output_file.append([])

    output_file.append(["ESTIMATED VALUES"])
    output_file.append(["-------------------------------------------"])
    output_file.append([])

    output_file.append(["Exterior Orientation Parameters"])
    output_file.append(["PointID\tImageID\tX_c\t\t\tY_c\t\t\tZ_c\t\t\tOmega\t\t\tPhi\t\t\tKappa"])

    for i in range(0, len(extmat)):
        # create temp vec too append to exte_ite
        exte_temp = []

        # ensure same imageID and cameraID as original file
        exte_temp.append(extmat[i][0])
        exte_temp.append("\t")
        exte_temp.append(extmat[i][1])
        exte_temp.append("\t")

        # add corrected exte values
        exte_temp.append(xhat[6 * i])
        exte_temp.append("\t")
        exte_temp.append(xhat[6 * i + 1])
        exte_temp.append("\t")
        exte_temp.append(xhat[6 * i + 2])
        exte_temp.append("\t")
        exte_temp.append(xhat[6 * i + 3])
        exte_temp.append("\t")
        exte_temp.append(xhat[6 * i + 4])
        exte_temp.append("\t")
        exte_temp.append(xhat[6 * i + 5])

        output_file.append(exte_temp)

    output_file.append(["-------------------------------------------"])
    output_file.append([])

    output_file.append(["Object Point Coordinates"])
    output_file.append(["PointID\t\tc\t\t\tY\t\t\tZ"])

    for i in range(0, len(objmat)):
        # create temp vec too append to obj_ite
        obj_temp = []

        # ensure same pointID as original file
        obj_temp.append(objmat[i][0])
        obj_temp.append("\t\t")

        # add corrected obj values
        obj_temp.append(xhat[len(extmat) * 6 + 3 * i])
        obj_temp.append("\t")
        obj_temp.append(xhat[len(extmat) * 6 + 3 * i + 1])
        obj_temp.append("\t")
        obj_temp.append(xhat[len(extmat) * 6 + 3 * i + 2])

        output_file.append(obj_temp)

    f = open("Formatted Bundle Adjustment Output File.txt", 'w')
    with f:
        write = csv.writer(f, delimiter=' ')
        write.writerows(output_file)
    f.close()