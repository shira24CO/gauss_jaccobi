All_Elementary_matrix = {} #stores all the elementary matrix
f = "matrix"

try:
    with open(f, "w") as outfile:
        def printmat(matrix):
            outfile.write("\n")
            for i in matrix:
                for j in i:
                    outfile.write(f'{j}  ')
                outfile.write('\n')
            outfile.write('\n')


        MAT = [[-1, -2, 5], [4, -1, 1], [1, 6, 2]]
        b = [[2], [4], [9]]

        def minor_of_an_item_in_the_matrix(mat, line, column):
            temp_matrix = []
            for my_line in range(len(mat)):
                matrix_line = []
                for my_column in range(len(mat)):
                    if my_line != line and my_column != column:
                        matrix_line.append(mat[my_line][my_column])
                temp_matrix.append(matrix_line)
            return list(filter(lambda x: x != [], temp_matrix))

        def find_determinanta_of_matrix(matrix):
            determinanta = 0
            sign = -1
            line = 0
            column = 0
            if len(matrix) == 2:
               return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            for line_value in matrix[0]:
                determinanta += pow(sign, (line+column)) * line_value * find_determinanta_of_matrix(minor_of_an_item_in_the_matrix(matrix, line, column))
                column += 1
            return determinanta

        def product_calculation(vector_line, vector_column): #Multiply line and column
            result = 0
            for element in range(len(vector_line)):
                result += vector_line[element]*vector_column[element]
            return result

        def swap_lines_of_matrix(matrix,index_line1,index_line2):
            for column in range(len(matrix)):
                temp_value = matrix[index_line2][column]
                matrix[index_line2][column] = matrix[index_line1][column]
                matrix[index_line1][column] = temp_value

        def check_column_in_matrix(matrix,some_col):
            for line in range(len(matrix)):
                if matrix[line][some_col] !=0:
                    return True # The column is not full of 0's
            return False # The column is full of 0's

        def swap_columns_of_matrix(matrix,index_column1): #'matrix' is a matrix that has a column full of 0's.
            last_column = len(matrix) - 1
            for column in range(last_column,index_column1,1):
                if check_column_in_matrix(matrix,column) == True:
                    for line in range(len(matrix)):
                        matrix[line][index_column1] = matrix[line][last_column]
                    for index in range(len(matrix)):
                        matrix[index][last_column] = 0
                return

        def multiply_matrix(mat1,mat2): # matrixes of n*n
            new_mat = []  # contains the result of the multiplication
            line_index = 0
            vector_col = []
            for line_mat1 in mat1:
                temp_line_mat = []
                for column in range(len(mat2[0])):
                    vector_col = []
                    for line in range(len(mat1)):
                        vector_col.append(mat2[line][column])
                    temp_line_mat.append(product_calculation(line_mat1,vector_col))
                new_mat.append(temp_line_mat)
            return new_mat


        def create_I_matrix(size):
            matrixI =[]
            for i in range(size):
                matrixI_helper = []
                for j in range(size):
                    if i == j:
                        matrixI_helper.append(1)
                    else:
                        matrixI_helper.append(0)
                matrixI.append(matrixI_helper)
            return matrixI

        def find_max_of_column(matrix,j):
            element = create_I_matrix(len(matrix))
            #  Find the maximun value in a column in order to change lines
            maxElem = abs(matrix[j][j])
            maxRow = j
            for k in range(j+1,len(matrix)):  # Interacting over the next line,in the same column
                if (abs(matrix[k][j]) > maxElem):
                    maxElem = abs(matrix[k][j])  # Next line on the diagonal
                    maxRow = k
            swap_lines_of_matrix(element,maxRow,j)
            return element

        def print_state(elementary,matrix):

            outfile.write("elementary:   ")
            outfile.write("\n")
            printmat(elementary)
            outfile.write("\n")
            outfile.write("matrix:   \n")
            outfile.write("")
            printmat(matrix)
            outfile.write("\n")
            outfile.write("result:   ")
            outfile.write("\n")
            result = multiply_matrix(elementary,matrix)
            printmat(result)
            outfile.write("\n")
            return result

        def elementary_matrix(matrix,result_vector):
            counter_for_elementary_matrix = 0
            counter_for_elementary_operations1 = (pow(len(matrix),2) + len(matrix) ) / 2 # In order to create an upper triangular form for matrix 3 *3 we operate 3+2+1 operations(sum of arithmetic progression)
            while counter_for_elementary_matrix != counter_for_elementary_operations1:
                for column in range(len(matrix)):#
                    elementary_mat = find_max_of_column(matrix,column)
                    #matrix = print_state(elementary_mat,matrix)
                   # result_vector = multiply_matrix(elementary_mat,result_vector)
                    for line in range(len(matrix)):
                        if line == column and matrix[line][column] !=0:
                            piv = 1 / matrix[line][column]
                            elementary_mat = create_I_matrix(len(matrix))
                            elementary_mat[line][column] = piv
                            #result_vector = multiply_matrix(elementary_mat,result_vector)
                            #matrix = print_state(elementary_mat, matrix)
                            counter_for_elementary_matrix += 1
                            All_Elementary_matrix[counter_for_elementary_matrix] = elementary_mat   # Enter new elementary matrix in the dictionary.
                        elif line == column and matrix[line][column] == 0: # we need to swap lines
                            line_to_swap_with = -1
                            for l in range(len(matrix)):
                                if matrix[l][column] != 0:
                                    line_to_swap_with = l
                                    swap_lines_of_matrix(matrix,line_to_swap_with,line)
                            if line_to_swap_with == -1: # we did not find in the column 'column' a number different than zero. Therefore we can not swap lines. So,we will try to swap columns.
                                swap_columns_of_matrix(matrix,column)
                        elif line != column and line > column:
                            elementary_mat = create_I_matrix(len(matrix))
                            piv =- matrix[line][column]/matrix[column][column]
                            elementary_mat[line][column] = piv
                            #matrix = print_state(elementary_mat,matrix)
                            #result_vector = multiply_matrix(elementary_mat,result_vector)
                            counter_for_elementary_matrix += 1
                            All_Elementary_matrix[counter_for_elementary_matrix] = elementary_mat
            # Until here we receive an upper triangle matrix
            counter_for_elementary_operations2 = ((pow(len(matrix),2) + len(matrix) ) / 2) -len(matrix)
            counter_for_elementary_matrix2 = 0
            while counter_for_elementary_matrix2 != counter_for_elementary_operations2:
                for column in range(len(matrix)-1,-1,-1):
                    for line in range(column-1,-1,-1):
                        if line != column and line < column:
                            elementary_mat = create_I_matrix(len(matrix))
                            piv = - matrix[line][column] / matrix[column][column]
                            elementary_mat[line][column] = piv
                            #matrix = print_state(elementary_mat,matrix)
                           # result_vector = multiply_matrix(elementary_mat,result_vector)
                            counter_for_elementary_matrix2 += 1
                            All_Elementary_matrix[counter_for_elementary_matrix+counter_for_elementary_matrix2] = elementary_mat
            mat_i = create_I_matrix(len(matrix))
            mat = create_I_matrix(len(matrix))
            mat_w = create_I_matrix(len(matrix))
            for i in range(len(All_Elementary_matrix) - 1, 0, -1):
                mat_w = All_Elementary_matrix[i]
                mat = multiply_matrix(mat, mat_w)
            return mat
        outfile.write("Solution of the linear equations:")
            #printmat(result_vector)
        def zero_matrix(matrix):
            mat = []
            for i in range(len(matrix)):
                mat.append([0 for i in matrix])
            return mat
        def sum_matrix(A, B):
            AB = create_I_matrix(len(A))
            for i in range(len(A)):
                for j in range(len(B)):
                    AB[i][j] = A[i][j] + B[i][j]
            return AB

        def jacobbi(G, H):
            # Xr = GXr-1 + H
            geuss = [[0], [0], [0]]
            temp = 0
            while temp < 0.000001:
                geuss_next = [[temp], [temp], [temp]]
                geuss_next[temp][temp][temp] = sum_matrix(multiply_matrix(G, geuss), H)
                print(geuss_next)
            return geuss_next


        geuss = [[0], [0], [0]]
        D = create_I_matrix(len(MAT))
        U = zero_matrix(MAT)
        L = zero_matrix(MAT)
        for i in range(len(MAT)):
            D[i][i] = MAT[i][i]
            invert_D = elementary_matrix(D, b)
        for i in range(len(MAT)):
            for j in range(len(MAT)):
                if i < j:
                    U[i][j] = MAT[i][j]
                if i > j:
                    L[i][j] = MAT[i][j]
        LU = zero_matrix(MAT)
        for i in range(len(MAT)):
            for j in range(len(MAT)):
                LU[i][j] = L[i][j] + U[i][j]
        G = multiply_matrix(invert_D, LU)
        H = multiply_matrix(invert_D, b)
        print(sum_matrix([1, 2, 3], [4, 5, 6]))


except OSError as err:
        print(f"OS error not file not found: {err}.")
        raise

