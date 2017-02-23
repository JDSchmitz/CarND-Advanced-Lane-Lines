import matplotlib.pyplot as plt

def plot_images(left_images, right_images, save_filename='output.png', left_title='left', right_title='right' ):
    ''' Plot left and right images with titles
    '''

    plt.rcParams.update({'font.size': 8})
    f = plt.figure(figsize=(15,10))

    rows = len(left_images)
    img_i = 1
    for i in range(rows):
        # Left 
        plt.subplot(rows, 2, img_i)
        plt.imshow(left_images[i])
        plt.axis('on')
        if i == 0:
            plt.title(left_title) 
        img_i += 1
        

        # Right column
        plt.subplot(rows, 2, img_i)
        plt.imshow(right_images[i], cmap='gray')
        plt.axis('on')
        if i == 0: 
            plt.title(right_title)
        img_i += 1
        

    # save figure
    plt.savefig(save_filename, bbox_inches='tight')
    print('Saved figure in:', save_filename)

def plot_image(img, title='title', save_file='out.png'):
    ''' Plot and save the figure
    '''
    plt.rcParams.update({'font.size': 8})
    f = plt.figure(figsize=(10,6))

    plt.imshow(img)
    plt.axis('off')
    if title is not None: 
        plt.title(title)

    plt.savefig(save_file)
    
def show_image(fig, img, i, title=None, cmap=None):
    ''' Plot given img using the counter i
    ''' 
    a = fig.add_subplot(2, 2, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i+1
