########################################
#
# Nome: Ailton de Almeida Rangel Neto  
#
# Nome: Hermanyo Herick Alves Souza  
#
########################################


import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt


def imread(n_arq): #Q.2 
    
    return  mp.imread(n_arq,np.uint8)

def nchannels(image): #Q0.3 

    return 1 if (len(image.shape)) == 2 else image.shape[2]
 
def size(image): #Q0.4 
    
    return image.shape[1],image.shape[0]

def rgb2gray(image): #Q0.5  
    if(nchannels(image)==1):
      return image
    
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]   
    return np.uint8((0.299 * r) + (0.587 * g) + (0.114 * b) / (0.299+0.587+0.114))

def imreadgray(n_arq): #Q0.6 
    
    image = imread(n_arq)

    if( nchannels(image) == 1 ):
        return image
    elif(nchannels(image) >= 3): 
        return rgb2gray(image)

def imshow(img): #Q0.7
    
    if(nchannels(img) == 1):
        plt.imshow(img, cmap = plt.get_cmap('gray'), interpolation = 'nearest')
    else:
        plt.imshow(img.astype('uint8'), interpolation = 'nearest')
        
    plt.show()

def thresh(image, val): #Q0.8
    
    temp_image = np.copy(image) 
    
    if(nchannels(temp_image) == 1):

        for i in range(len(temp_image)):
            for j in range (len(temp_image[i])):
                if(temp_image[i][j] >= val):
                    temp_image[i][j] = 255
                else:
                    temp_image[i][j] = 0

    elif( nchannels(temp_image) == 3 ):

        for i in range(len(temp_image)):
            for j in range (len(temp_image[i])):
                for k in range(len(temp_image[i][j])):
                    if(temp_image[i][j][k] >= val):
                        temp_image[i][j][k] = 255
                    else:
                        temp_image[i][j][k] = 0 

    return temp_image

def negative(image_org): #Q0.9
    image = np.copy(image_org)  
    if( nchannels(image)==1 ):

        for i in range(len(image)):
            for j in range (len(image[i])):
                image[i][j] *= -1 

    elif(nchannels(image)==3):  
        for i in range(len(image)):
            for j in range (len(image[i])):
                for k in range(len(image[i][j])):
                    image[i][j][k] *= -1
    return image

def contrast(f,r,m): #Q0.10   
    img = np.copy(f)

    if( nchannels(img)==1 ):
        for i in range(len(img)):
            for j in range (len(img[i])):
                a = img[i][j]
                a = r*(a-m) + m 
                if(a < 0): a = 0
                elif(a > 255): a = 255   
                img[i][j] = a

    elif(nchannels(img)==3):
        for i in range(len(img)):
            for j in range (len(img[i])):
                for k in range(len(img[i][j])): 
                    a = img[i][j][k]
                    a = r*(a-m) + m 
                    if(a < 0): a = 0
                    elif(a > 255): a = 255   
                    img[i][j][k] = a
                 
    return img

def hist(image): #Q0.11 
    
    if( nchannels(image)==1 ):
        
        histo = [0]*256

        for i in range(len(image)):
            for j in range (len(image[i])):
                histo[ image[i][j] ] += 1 
        return histo

    elif(nchannels(image)==3):

        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        histo = [[0]]*3
        for i in range(3):
            histo[i] = [0]*256        

        for i in range(len(r)):
            for j in range (len(r[i])):
                histo[0][ r[i][j] ] += 1
                histo[1][ g[i][j] ] += 1
                histo[2][ b[i][j] ] += 1
    
    
    return histo

def showhist(histo, bin = 1): #Q0.12
    
    if(len(histo) == 3):
        
        n = bin
        j = 0 
        qtd = len(histo[0])//n + len(histo[0])%n

        valores = [[0]]*3   
        for i in range(3):
            valores[i] = [0]*qtd
        if(bin == 1 ):
            r,g,b = histo[0],histo[1],histo[2]
            
        else:
            for i in range(qtd):      
                valores[0][i] =  sum(histo[0][j:n])
                valores[1][i] =  sum(histo[1][j:n])
                valores[2][i] =  sum(histo[2][j:n])
                j = n
                n += bin

            r,g,b = valores[0],valores[1],valores[2]
        
        x_pos = np.arange(len(g))
        
        width_n = 0.2
        bar_larg = 0.2
        plt.figure(figsize =(15,10))     
             
    
        plt.bar(x_pos, r, width=width_n, color='r')
    
        plt.bar(x_pos + bar_larg , g , width=width_n, color='g')
    
        plt.bar(x_pos + bar_larg + bar_larg, b , width=width_n, color='b')
        
        plt.xticks(x_pos + bar_larg, x_pos )
        
        plt.title('HISTOGRAMA')
        plt.ylabel('QUANTIDADE')
        plt.xlabel('INTENSIDADE')        
        plt.show()
        
    else:
        
        n = bin
        j = 0
        qtd = np.uint8(len(histo)/n + len(histo)%n)
        valores = [0]*qtd
        if(bin == 1 ):

            valores = np.copy(histo)
        else:
            for i in range(qtd):      
                valores[i] =  sum(histo[j:n])
                j = n
                n += bin         
               
        y_axis = valores
        x_axis = range(len(y_axis))

        width_n = 0.2
        bar_color = 'blue'
        plt.figure(figsize =(15,10))

         
        plt.bar(x_axis, y_axis, width=width_n, color=bar_color, align='center')
        plt.xticks(x_axis , x_axis )
        
        plt.title('HISTOGRAMA')
        plt.ylabel('QUANTIDADE')
        plt.xlabel('INTENSIDADE')
        plt.show()
 
def histeq(image): #Q.14
    h = hist(image)
    H = np.cumsum(h) / float(np.sum(h))
    e = np.floor(H[np.uint8(image)]*255)
    return e.reshape(image.shape).astype('uint8')

def convolve(image, mask): #Q0.15 
    
    g = np.copy(image)  
    if(nchannels(image)==1):
        for i in range(len(image)):
            for j in range (len(image[i])):
                g[i][j] = 0
                for k in range (len(mask)): 
                    for t in range(len(mask[k])):    
                        g[i][j] += image[min(max(i+k-1, 0), image.shape[0]-1)][min(max(j+t-1, 0), image.shape[1]-1)]*mask[k][t]

    elif(nchannels(image)==3):   
        for i in range(len(image)):
            for j in range (len(image[i])):   
                    g[i][j][0] = g[i][j][1] = g[i][j][2] = 0
                    for k in range (len(mask)): 
                        for t in range(len(mask[k])):  
                            f = image[min(max(i+k-1, 0), image.shape[0]-1)][min(max(j+t-1, 0), image.shape[1]-1)]
                            g[i][j][0] += f[0]*mask[k][t]
                            g[i][j][1] += f[1]*mask[k][t] 
                            g[i][j][2] += f[2]*mask[k][t]
    
    return g                  

def maskBlur(): #Q0.16   
    return np.dot([[1, 2, 1], [2, 4, 2], [1,2,1]],1/16) 

def blur(image): #Q0.17  
    return convolve(image,maskBlur())

def seSquare3(): #Q0.18
    return np.dot([[1,1, 1], [1, 1, 1], [1, 1, 1]],1)

def seCross3(): #Q0.19
    return np.dot([[0, 1,0], [1, 1, 1], [0, 1, 0]],1)

def erode(image,eleBin): #Q0.20
    
    n = len(eleBin)
    m = len(eleBin[0])
   
    menorimg = np.copy(image)
    

    a = ( n-1 ) // 2
    b = ( m-1 ) // 2
    if(nchannels(image)==1):
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = []
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):

                                menor.append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)])
                
                menorimg[i][j] = min(menor)
   
    if(nchannels(image)==3):
        
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = [[],[],[]]
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):

                                menor[0].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][0])
                                
                                menor[1].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][1])
                                
                                menor[2].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][2])

                menorimg[i][j][0] = min(menor[0])
                menorimg[i][j][1] = min(menor[1])
                menorimg[i][j][2] = min(menor[2])


    return menorimg

def dilate(image,eleBin): #Q0.21
    
    n = len(eleBin)
    m = len(eleBin[0])
   
    menorimg = np.copy(image)
    

    a = ( n-1 ) // 2
    b = ( m-1 ) // 2
    if(nchannels(image)==1):
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = []
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):

                                menor.append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)])
                
                menorimg[i][j] = max(menor)
   
    if(nchannels(image)==3):
        
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = [[],[],[]]
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):
                                menor[0].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][0])
                                menor[1].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][1])            
                                menor[2].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][2])

                menorimg[i][j][0] = max(menor[0])
                menorimg[i][j][1] = max(menor[1])
                menorimg[i][j][2] = max(menor[2])


    return menorimg
