from __future__ import print_function, absolute_import
from PIL import Image
import matplotlib.pyplot as plt



def visualize_results(indices, df,image, k=3, width=224, height=224, save_dir='',set=''):
    '''
    Args:
        indices: indices sorted by similarity to the query image
        df: the dataframe
        image: query image
        k: number of similar images desired as output
        width: width of each plotted image
        height: height of each plotted image
        save_dir: directory where to save the plot
    Returns:
        None
    '''
    #create the plot
    plt.figure(figsize=(width*(k+1)/100, height/100))
    plt.subplot(1,k+1,1)
    plt.title('Query')
    #image=image.resize((width,height))
    plt.imshow(image)
    plt.axis('off')
    
    
    for i in range(k):
        qimg_row = df.iloc[int(indices[0][i])]
        
        qimg_cell_id = qimg_row['TEMPLATE_ID']
        # print('ciao')
        qimg_filename = qimg_row['PATH']
        qimg = Image.open(qimg_filename)
        qimg = qimg.convert("RGB")
        qimg = qimg.resize((width, height))
        #subplot the i-th match
        plt.subplot(1,k+1,i+2)
        plt.title(qimg_cell_id,fontdict={'fontsize': 6})
        plt.imshow(qimg)
        plt.axis('off')
    plt.tight_layout()
    #save the plot in save_dir
    name = f'{save_dir}.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close()
   