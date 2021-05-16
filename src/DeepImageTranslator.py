from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import ctypes
import ModelCreator
import ImageGEN
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os


ctypes.windll.shcore.SetProcessDpiAwareness(1)



############
############
# Command: Load input images
def Load_train_input():
    global Train_input_dir_list, Train_index_list, Img1, cumulative_train_index, image_set, o_h, o_w, o_c
    # Create list of input images
    Train_input_dir_list = filedialog.askopenfilenames(parent=root, title='Select input image files for training')
    Train_index_list = np.arange(len(Train_input_dir_list))
    cumulative_train_index = 0
    Img1 = Image.open(Train_input_dir_list[0])
    shape = cv2.imread(Train_input_dir_list[0])
    o_h = shape.shape[0]
    o_w = shape.shape[1]
    o_c = shape.shape[2]
    Image_panel.add(Image1)
    image_set = 1

# Command: Load target images
def Load_train_target():
    global Train_target_dir_list, Train_index_list, Img2, cumulative_train_index, image_set, t_h, t_w, t_c
    # Create list of input images
    Train_target_dir_list = filedialog.askopenfilenames(parent=root,title='Select target image files for training')
    Train_index_list = np.arange(len(Train_target_dir_list))
    cumulative_train_index = 0
    Img2 = Image.open(Train_target_dir_list[0])
    shape = cv2.imread(Train_target_dir_list[0])
    t_h = shape.shape[0]
    t_w = shape.shape[1]
    t_c = shape.shape[2]
    Image_panel.add(Image2)
    image_set = 1

def Load_val_input():
    global Val_input_dir_list, Val_index_list, Img1, cumulative_val_index, image_set
    # Create list of input images
    Val_input_dir_list = filedialog.askopenfilenames(parent=root, title='Select input image files for validation')
    Val_index_list = np.arange(len(Val_input_dir_list))
    cumulative_val_index = 0
    Img1 = Image.open(Val_input_dir_list[0])
    Image_panel.add(Image1)
    image_set = 2

# Command: Load target images
def Load_val_target():
    global Val_target_dir_list, Val_index_list, Img2, cumulative_val_index, image_set
    # Create list of input images
    Val_target_dir_list = filedialog.askopenfilenames(parent=root,title='Select target image files for validation')
    Val_index_list = np.arange(len(Val_target_dir_list))
    cumulative_val_index = 0
    Img2 = Image.open(Val_target_dir_list[0])
    Image_panel.add(Image2)
    image_set = 2

def Load_trans_input():
    global Trans_input_dir_list, Trans_index_list, Img1, cumulative_Trans_index, image_set
    # Create list of input images
    Trans_input_dir_list = filedialog.askopenfilenames(parent=root, title='Select input image files for translation')
    Trans_index_list = np.arange(len(Trans_input_dir_list))
    cumulative_Trans_index = 0
    Img1 = Image.open(Trans_input_dir_list[0])
    Image_panel.add(Image1)
    image_set = 3




# Command: Define new model window
def Open_new_model_window():
    global o_w, o_h, o_c, t_w, t_h, t_c, model_type, num_layers, initial_ch, gan_onoff, deep_sup, current_model

    New_model_window = Toplevel()
    New_model_window.title('Define new model')
    New_model_window.iconphoto(False, photo)
    New_model_window.geometry("400x500")

    New_model_panel = PanedWindow(New_model_window, orient=HORIZONTAL, sashwidth=20, bg='#4f4f4f')
    New_model_panel.pack(fill=BOTH, expand=1)

    New_model_left_panel = Frame(New_model_panel, bg='#4f4f4f', width=300)
    New_model_panel.add(New_model_left_panel)

    Leftborder = Frame(New_model_left_panel, bg='#4f4f4f', width=10)
    Leftborder.pack(side=LEFT, fill=BOTH, expand=0)

    New_model_inputs = Frame(New_model_left_panel, bg='#4f4f4f', width=190)
    New_model_inputs.pack(side=LEFT, fill=BOTH, expand=0)

    New_model_inputs_R1 = Frame(New_model_inputs, height=20, bg='#4f4f4f')
    New_model_inputs_R1.pack(side=TOP, fill=BOTH, expand=0)

    New_model_inputs_R2 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_R2.pack(side=TOP, fill=BOTH, expand=0)
    New_model_Ltext = Label(New_model_inputs_R2, text="Input image size", font=('helvetica', 9, 'bold'), anchor=W, bg='#4f4f4f', fg='#ffffff')
    New_model_Ltext.pack(side=LEFT, expand=1, fill=X)

    New_model_inputs_R3 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_R3.pack(side=TOP, fill=BOTH, expand=0)
    New_model_Linputs = Frame(New_model_inputs_R3, height=50, bg='#4f4f4f')
    New_model_Linputs.pack(side=LEFT, expand=0, fill=X)
    L_filler1 = Label(New_model_Linputs, width=3, bg='#4f4f4f')
    L_filler1.pack(side=LEFT, expand=0, fill=X)
    O_W = Entry(New_model_Linputs, width=7, bg="#1f1f1f", fg='#ffffff')
    O_W.insert(0, o_w)
    O_W.pack(side=LEFT, expand=0, fill=X)
    O_H = Entry(New_model_Linputs, width=7, bg="#1f1f1f", fg='#ffffff')
    O_H.insert(0, o_h)
    O_H.pack(side=LEFT, expand=0, fill=X)
    O_C = Entry(New_model_Linputs, width=7, bg="#1f1f1f", fg='#ffffff')
    O_C.insert(0, o_c)
    O_C.pack(side=LEFT, expand=0, fill=X)


    New_model_inputs_filler5 = Frame(New_model_inputs, height=30, bg='#4f4f4f')
    New_model_inputs_filler5.pack(side=TOP, fill=BOTH, expand=0)

    New_model_Rtext = Label(New_model_inputs, text="Target image size", font=('helvetica', 9, 'bold'), bg='#4f4f4f', fg='#ffffff', anchor=W)
    New_model_Rtext.pack(side=TOP, expand=0, fill=X)

    New_model_Rinputs = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_Rinputs.pack(side=TOP, expand=0, fill=X)
    R_filler1 = Label(New_model_Rinputs, width=3, bg='#4f4f4f')
    R_filler1.pack(side=LEFT, expand=0, fill=X)
    T_W = Entry(New_model_Rinputs, width=7, bg="#1f1f1f", fg='#ffffff')
    T_W.insert(0, t_w)
    T_W.pack(side=LEFT, expand=0, fill=X)
    T_H = Entry(New_model_Rinputs, width=7, bg="#1f1f1f", fg='#ffffff')
    T_H.insert(0, t_h)
    T_H.pack(side=LEFT, expand=0, fill=X)
    T_C = Entry(New_model_Rinputs, width=7, bg="#1f1f1f", fg='#ffffff')
    T_C.insert(0, t_c)
    T_C.pack(side=LEFT, expand=0, fill=X)


    New_model_inputs_filler4 = Frame(New_model_inputs, height=30, bg='#4f4f4f')
    New_model_inputs_filler4.pack(side=TOP, fill=BOTH, expand=0)

    New_model_inputs_R4 = Frame(New_model_inputs, height=30, bg='#4f4f4f')
    New_model_inputs_R4.pack(side=TOP, expand=0, fill=BOTH)
    Model_type_box = Frame(New_model_inputs_R4, bg='#4f4f4f', height=30)
    Model_type_box.pack(side=LEFT, expand=1, fill=X)

    Model_Name = StringVar()
    Model_Name.set("U-Net")

    Model_Type = OptionMenu(Model_type_box, Model_Name, "Model type", "U-Net")
    Model_Type.config(bg="#1f1f1f", fg="#ffffff", bd=0, relief=FLAT, activeforeground='#ffffff', activebackground='#1f1f1f', width=12, highlightthickness=0)
    Model_Type["menu"].config(bg="#1f1f1f", fg="#ffffff", bd=0, activeforeground='#ffffff', activebackground='#4f4f4f')
    Model_Type.pack(side=LEFT, expand=0, fill=X)


    New_model_inputs_filler34 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_filler34.pack(side=TOP, fill=BOTH, expand=0)

    Num_of_layers_box = Frame(New_model_inputs, bg='#4f4f4f', height=30)
    Num_of_layers_box.pack(side=TOP, expand=0, fill=BOTH)
    Num_of_layers_text = Label(Num_of_layers_box, text="Number of layers/blocks", bg='#4f4f4f', fg='#ffffff')
    Num_of_layers_text.pack(side=LEFT, expand=0, fill=X)
    Num_of_layers_filler = Label(Num_of_layers_box, width=1, bg='#4f4f4f')
    Num_of_layers_filler.pack(side=LEFT, expand=0, fill=X)
    Num_of_layers_entry = Entry(Num_of_layers_box, width=7, fg='#ffffff', bg="#1f1f1f")
    Num_of_layers_entry.insert(0, "5")
    Num_of_layers_entry.pack(side=LEFT, expand=0, fill=X)

    Num_of_channels_box = Frame(New_model_inputs, bg='#4f4f4f', height=30)
    Num_of_channels_box.pack(side=TOP, expand=0, fill=BOTH)
    Num_of_channels_text = Label(Num_of_channels_box, text="Number of channels of the first convolution", bg='#4f4f4f', fg='#ffffff')
    Num_of_channels_text.pack(side=LEFT, expand=0, fill=X)
    Num_of_channels_filler = Label(Num_of_channels_box, width=1, bg='#4f4f4f')
    Num_of_channels_filler.pack(side=LEFT, expand=0, fill=X)
    Num_of_channels_entry = Entry(Num_of_channels_box, width=7, fg='#ffffff', bg="#1f1f1f")
    Num_of_channels_entry.insert(0, "16")
    Num_of_channels_entry.pack(side=LEFT, expand=0, fill=X)


    New_model_inputs_filler3 = Frame(New_model_inputs, height=20, bg='#4f4f4f')
    New_model_inputs_filler3.pack(side=TOP, fill=BOTH, expand=0)


    New_model_inputs_R6 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_R6.pack(side=TOP, fill=BOTH, expand=0)

    Deep_Sup = IntVar()
    Deep_Sup_Check = Checkbutton(New_model_inputs_R6, text="Deep supervision (for U-Net)", variable=Deep_Sup, anchor=W, bg='#4f4f4f', fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff', activebackground='#4f4f4f')
    Deep_Sup_Check.pack(side=LEFT)

    New_model_inputs_filler2 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_filler2.pack(side=TOP, fill=BOTH, expand=1)

    New_model_inputs_R7 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_R7.pack(side=TOP, fill=BOTH, expand=0)

    def get_var_gen_model():
        global o_w, o_h, o_c, t_w, t_h, t_c, model_type, num_layers, initial_ch, gan_onoff, deep_sup, current_model
        o_w = int(O_W.get())
        o_h = int(O_H.get())
        o_c = int(O_C.get())

        t_w = int(T_W.get())
        t_h = int(T_H.get())
        t_c = int(T_C.get())

        model_type = Model_Name.get()
        num_layers = int(Num_of_layers_entry.get())
        initial_ch = int(Num_of_channels_entry.get())

        gan_onoff = 0
        deep_sup = int(Deep_Sup.get())

        current_model = ModelCreator.modelcreator(o_w, o_h, o_c, t_w, t_h, t_c, model_type, num_layers, initial_ch,
                                                  deep_sup, gan_onoff)
        current_model.compile(optimizer=opt_type, loss=_loss, metrics=[psnr, ssim, "mse", "mae"])
        current_model.summary()

        # weight_path = Train_input_dir_list[0]
        # weight_folder_path = os.path.dirname(weight_path)
        #
        # filepath = os.path.join(weight_folder_path, "model structure.png")
        # tf.keras.utils.plot_model(current_model, to_file=filepath, show_shapes=True)


    Create_Button = Button(New_model_inputs_R7, text="Create model", relief=FLAT, overrelief=FLAT, bg='#1f1f1f',
                           fg='#ffffff', activebackground='#4f4f4f', activeforeground='#ffffff',
                           command=get_var_gen_model)
    Create_Button.pack()

    New_model_inputs_filler1 = Frame(New_model_inputs, height=50, bg='#4f4f4f')
    New_model_inputs_filler1.pack(side=TOP, fill=BOTH, expand=0)


def Apply_augmentation():
    global rp_prob, in_rp, tar_rp,  s_p, in_s, tar_s, r_d, in_r, tar_r, t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, \
        b_p, in_b, tar_b,  n_p, in_n, tar_n,  b_u, in_br, tar_br, c_u, in_c, tar_c


    rp_prob = float(Rp_Prob.get())
    in_rp = int(In_Rp.get())
    tar_rp = int(Tar_Rp.get())

    s_p = int(S_P.get())
    r_d = int(R_D.get())
    t_p = int(T_P.get())

    in_s = int(In_S.get())
    tar_s = int(Tar_S.get())

    in_r = int(In_R.get())
    tar_r = int(Tar_R.get())

    in_t = int(In_T.get())
    tar_t = int(Tar_T.get())

    in_f = int(In_F.get())
    tar_f = int(Tar_F.get())

    d_n = int(D_N.get())
    d_s = int(D_S.get())
    in_d = int(In_D.get())
    tar_d = int(Tar_D.get())

    e_p = int(E_P.get())
    in_e = int(In_E.get())
    tar_e = int(Tar_E.get())

    b_p = int(B_P.get())
    in_b = int(In_B.get())
    tar_b = int(Tar_B.get())

    n_p = float(N_P.get())
    in_n = int(In_N.get())
    tar_n = int(Tar_N.get())

    b_u = float(B_U.get())
    in_br = int(In_Br.get())
    tar_br = int(Tar_Br.get())

    c_u = float(C_U.get())
    in_c = int(In_C.get())
    tar_c = int(Tar_C.get())

    switch_to_train()



def Open_data_aug_window():
    global Rp_Prob, In_Rp, Tar_Rp,  S_P, In_S, Tar_S,  R_D, In_R, Tar_R,  T_P, In_T, Tar_T,  D_N, D_S, In_F, Tar_F,  In_D, Tar_D,  E_P, In_E, Tar_E, \
        B_P, In_B, Tar_B,  N_P, In_N, Tar_N,  B_U, In_Br, Tar_Br,  C_U, In_C, Tar_C

    data_aug_window = Toplevel(bg='#4f4f4f')
    data_aug_window.title('Choose data augmentation schemes')
    data_aug_window.iconphoto(False, photo)
    data_aug_window.geometry("900x550")

    data_aug_window_bg = Frame(data_aug_window, bg='#4f4f4f', width=900)
    data_aug_window_bg.pack(side=LEFT, expand=1, fill=BOTH, anchor=NW, padx=10, pady=10)

    data_aug_window_panel = Frame(data_aug_window_bg, bg='#4f4f4f', width=880, height=450)
    data_aug_window_panel.pack(side=TOP, expand=1, anchor=NW)
    data_aug_window_panel.pack_propagate(0)

    data_aug_Col1 = Frame(data_aug_window_panel, bg='#4f4f4f', width=680)
    data_aug_Col1.pack(side=LEFT, expand=1, fill=Y)
    data_aug_Col1.pack_propagate(0)
    data_aug_c1_r1 = Label(data_aug_Col1, text="Apply random:", font=('helvetica', 9, 'bold'), bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r1.pack(side=TOP, anchor=NW)
    data_aug_c1_r2 = Label(data_aug_Col1, bg='#4f4f4f')
    data_aug_c1_r2.pack(side=TOP, anchor=NW)

    data_aug_c1_r14 = Frame(data_aug_Col1)
    data_aug_c1_r14.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r14_1 = Label(data_aug_c1_r14, text="- Ripple effect with : ", bg='#4f4f4f', fg='#ffffff', height=1)
    data_aug_c1_r14_1.pack(side=LEFT, anchor=W)
    Rp_Prob = Entry(data_aug_c1_r14, width=7, bg="#1f1f1f", fg='#ffffff')
    Rp_Prob.insert(0, "0.7")
    Rp_Prob.pack(side=LEFT, expand=0, anchor=W)
    data_aug_c1_r3_14 = Label(data_aug_c1_r14, text="probability", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r3_14.pack(side=LEFT, anchor=W)

    data_aug_c1_r3 = Frame(data_aug_Col1)
    data_aug_c1_r3.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r3_1 = Label(data_aug_c1_r3, text="- Scaling ", bg='#4f4f4f', fg='#ffffff', height=1)
    data_aug_c1_r3_1.pack(side=LEFT, anchor=W)
    S_P = Entry(data_aug_c1_r3, width=7, bg="#1f1f1f", fg='#ffffff')
    S_P.insert(0, "30")
    S_P.pack(side=LEFT, expand=0, anchor=W)
    data_aug_c1_r3_3 = Label(data_aug_c1_r3, text="pixels", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r3_3.pack(side=LEFT, anchor=W)

    data_aug_c1_r4 = Frame(data_aug_Col1)
    data_aug_c1_r4.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r4_1 = Label(data_aug_c1_r4, text="- Rotation ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r4_1.pack(side=LEFT, anchor=W)
    R_D = Entry(data_aug_c1_r4, width=7, bg="#1f1f1f", fg='#ffffff')
    R_D.insert(0, "70")
    R_D.pack(side=LEFT, anchor=W)
    data_aug_c1_r4_3 = Label(data_aug_c1_r4, text="degrees", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r4_3.pack(side=LEFT, anchor=W)

    data_aug_c1_r5 = Frame(data_aug_Col1)
    data_aug_c1_r5.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r5_1 = Label(data_aug_c1_r5, text="- Translation ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r5_1.pack(side=LEFT, anchor=W)
    T_P = Entry(data_aug_c1_r5, width=7, bg="#1f1f1f", fg='#ffffff')
    T_P.insert(0, "30")
    T_P.pack(side=LEFT, anchor=W)
    data_aug_c1_r5_3 = Label(data_aug_c1_r5, text="degrees", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r5_3.pack(side=LEFT, anchor=W)

    data_aug_c1_r6 = Label(data_aug_Col1, text="- Flipping ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r6.pack(side=TOP, anchor=W, expand=1)

    data_aug_c1_r7 = Frame(data_aug_Col1)
    data_aug_c1_r7.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r7_1 = Label(data_aug_c1_r7, text="- Drop up to ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r7_1.pack(side=LEFT, anchor=W, expand=1)
    D_N = Entry(data_aug_c1_r7, width=7, bg="#1f1f1f", fg='#ffffff')
    D_N.insert(0, "12")
    D_N.pack(side=LEFT, anchor=W)
    data_aug_c1_r7_3 = Label(data_aug_c1_r7, text="rectangles of", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r7_3.pack(side=LEFT, anchor=W)
    D_S = Entry(data_aug_c1_r7, width=7, bg="#1f1f1f", fg='#ffffff')
    D_S.insert(0, "20")
    D_S.pack(side=LEFT, anchor=W)
    data_aug_c1_r7_4 = Label(data_aug_c1_r7, text="pixels", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r7_4.pack(side=LEFT, anchor=W)

    data_aug_c1_r8 = Frame(data_aug_Col1)
    data_aug_c1_r8.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r8_1 = Label(data_aug_c1_r8, text="- Elastic transform with magnitude of", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r8_1.pack(side=LEFT, anchor=W, expand=1)
    E_P = Entry(data_aug_c1_r8, width=7, bg="#1f1f1f", fg='#ffffff')
    E_P.insert(0, "2")
    E_P.pack(side=LEFT, anchor=W)
    data_aug_c1_r8_3 = Label(data_aug_c1_r8, text="pixels", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r8_3.pack(side=LEFT, anchor=W)

    data_aug_c1_r9 = Frame(data_aug_Col1)
    data_aug_c1_r9.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r9_1 = Label(data_aug_c1_r9, text="- Gaussian blur with radius of ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r9_1.pack(side=LEFT, anchor=W, expand=1)
    B_P = Entry(data_aug_c1_r9, width=7, bg="#1f1f1f", fg='#ffffff')
    B_P.insert(0, "3")
    B_P.pack(side=LEFT, anchor=W)
    data_aug_c1_r9_3 = Label(data_aug_c1_r9, text="pixels", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r9_3.pack(side=LEFT, anchor=W)

    data_aug_c1_r10 = Frame(data_aug_Col1)
    data_aug_c1_r10.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r10_1 = Label(data_aug_c1_r10, text="- Gaussian noise with : ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r10_1.pack(side=LEFT, anchor=W, expand=1)
    N_P = Entry(data_aug_c1_r10, width=7, bg="#1f1f1f", fg='#ffffff')
    N_P.insert(0, "0.1")
    N_P.pack(side=LEFT, anchor=W)
    data_aug_c1_r10_3 = Label(data_aug_c1_r10, text="probability", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r10_3.pack(side=LEFT, anchor=W)


    data_aug_c1_r12 = Frame(data_aug_Col1)
    data_aug_c1_r12.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r12_1 = Label(data_aug_c1_r12, text="- Brightness ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r12_1.pack(side=LEFT, anchor=W)
    B_U = Entry(data_aug_c1_r12, width=7, bg="#1f1f1f", fg='#ffffff')
    B_U.insert(0, "1.5")
    B_U.pack(side=LEFT, anchor=W)
    data_aug_c1_r12_3 = Label(data_aug_c1_r12, text="units", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r12_3.pack(side=LEFT, anchor=W)

    data_aug_c1_r13 = Frame(data_aug_Col1)
    data_aug_c1_r13.pack(side=TOP, anchor=W, expand=1)
    data_aug_c1_r13_1 = Label(data_aug_c1_r13, text="- Contrast ", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r13_1.pack(side=LEFT, anchor=W)
    C_U = Entry(data_aug_c1_r13, width=7, bg="#1f1f1f", fg='#ffffff')
    C_U.insert(0, "1.5")
    C_U.pack(side=LEFT, anchor=W)
    data_aug_c1_r13_3 = Label(data_aug_c1_r13, text="units", bg='#4f4f4f', fg='#ffffff')
    data_aug_c1_r13_3.pack(side=LEFT, anchor=W)



    data_aug_Col2 = Frame(data_aug_window_panel, width=100, bg='#4f4f4f')
    data_aug_Col2.pack(side=LEFT, expand=1, fill=Y)
    data_aug_Col2.pack_propagate(0)
    data_aug_c2_r1 = Label(data_aug_Col2, text=" Apply to:", font=('helvetica', 9, 'bold'), bg='#4f4f4f', fg='#ffffff')
    data_aug_c2_r1.pack(side=TOP, anchor=NW)
    data_aug_c2_r2 = Label(data_aug_Col2, text="Input images", bg='#4f4f4f', fg='#ffffff')
    data_aug_c2_r2.pack(side=TOP)

    In_Rp = IntVar()
    data_aug_c2_r14 = Checkbutton(data_aug_Col2, variable=In_Rp, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r14.pack(side=TOP, expand=1)

    In_S = IntVar()
    data_aug_c2_r3 = Checkbutton(data_aug_Col2, variable=In_S, bg='#4f4f4f',
                fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff', activebackground='#4f4f4f')
    data_aug_c2_r3.select()
    data_aug_c2_r3.pack(side=TOP, expand=1)

    In_R = IntVar()
    data_aug_c2_r4 = Checkbutton(data_aug_Col2, variable=In_R, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r4.select()
    data_aug_c2_r4.pack(side=TOP, expand=1)

    In_T = IntVar()
    data_aug_c2_r5 = Checkbutton(data_aug_Col2, variable=In_T, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r5.select()
    data_aug_c2_r5.pack(side=TOP, expand=1)

    In_F = IntVar()
    data_aug_c2_r6 = Checkbutton(data_aug_Col2, variable=In_F, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r6.select()
    data_aug_c2_r6.pack(side=TOP, expand=1)

    In_D = IntVar()
    data_aug_c2_r7 = Checkbutton(data_aug_Col2, variable=In_D, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r7.pack(side=TOP, expand=1)

    In_E = IntVar()
    data_aug_c2_r8 = Checkbutton(data_aug_Col2, variable=In_E, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r8.pack(side=TOP, expand=1)

    In_B = IntVar()
    data_aug_c2_r9 = Checkbutton(data_aug_Col2, variable=In_B, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r9.pack(side=TOP, expand=1)

    In_N = IntVar()
    data_aug_c2_r10 = Checkbutton(data_aug_Col2, variable=In_N, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f')
    data_aug_c2_r10.pack(side=TOP, expand=1)


    In_Br = IntVar()
    data_aug_c2_r12 = Checkbutton(data_aug_Col2, variable=In_Br, bg='#4f4f4f',
                                  fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                  activebackground='#4f4f4f')
    data_aug_c2_r12.pack(side=TOP, expand=1)

    In_C = IntVar()
    data_aug_c2_r13 = Checkbutton(data_aug_Col2, variable=In_C, bg='#4f4f4f',
                                  fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                  activebackground='#4f4f4f')
    data_aug_c2_r13.pack(side=TOP, expand=1)




    data_aug_Col3 = Frame(data_aug_window_panel, width=100, bg='#4f4f4f')
    data_aug_Col3.pack(side=LEFT, expand=1, fill=Y)
    data_aug_Col3.pack_propagate(0)
    data_aug_c3_r1 = Label(data_aug_Col3, text=" ", font=('helvetica', 9, 'bold'), bg='#4f4f4f', fg='#ffffff')
    data_aug_c3_r1.pack(side=TOP, anchor=NW)
    data_aug_c3_r2 = Label(data_aug_Col3, text="Target images", bg='#4f4f4f', fg='#ffffff')
    data_aug_c3_r2.pack(side=TOP)

    Tar_Rp = IntVar()
    data_aug_c2_r14 = Checkbutton(data_aug_Col3, variable=Tar_Rp, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r14.pack(side=TOP, expand=1)

    Tar_S = IntVar()
    data_aug_c2_r3 = Checkbutton(data_aug_Col3, variable=Tar_S, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r3.select()
    data_aug_c2_r3.pack(side=TOP, expand=1)

    Tar_R = IntVar()
    data_aug_c2_r4 = Checkbutton(data_aug_Col3, variable=Tar_R, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r4.select()
    data_aug_c2_r4.pack(side=TOP, expand=1)

    Tar_T = IntVar()
    data_aug_c2_r5 = Checkbutton(data_aug_Col3, variable=Tar_T, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r5.select()
    data_aug_c2_r5.pack(side=TOP, expand=1)

    Tar_F = IntVar()
    data_aug_c2_r6 = Checkbutton(data_aug_Col3, variable=Tar_F, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r6.select()
    data_aug_c2_r6.pack(side=TOP, expand=1)

    Tar_D = IntVar()
    data_aug_c2_r7 = Checkbutton(data_aug_Col3, variable=Tar_D, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r7.pack(side=TOP, expand=1)

    Tar_E = IntVar()
    data_aug_c2_r8 = Checkbutton(data_aug_Col3, variable=Tar_E, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r8.pack(side=TOP, expand=1)

    Tar_B = IntVar()
    data_aug_c2_r9 = Checkbutton(data_aug_Col3, variable=Tar_B, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r9.pack(side=TOP, expand=1)

    Tar_N = IntVar()
    data_aug_c2_r10 = Checkbutton(data_aug_Col3, variable=Tar_N, bg='#4f4f4f',
                                 fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                 activebackground='#4f4f4f', height=1)
    data_aug_c2_r10.pack(side=TOP, expand=1)


    Tar_Br = IntVar()
    data_aug_c2_r12 = Checkbutton(data_aug_Col3, variable=Tar_Br, bg='#4f4f4f',
                                  fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                  activebackground='#4f4f4f', height=1)
    data_aug_c2_r12.pack(side=TOP, expand=1)

    Tar_C = IntVar()
    data_aug_c2_r13 = Checkbutton(data_aug_Col3, variable=Tar_C, bg='#4f4f4f',
                                  fg='#ffffff', selectcolor="#4f4f4f", activeforeground='#ffffff',
                                  activebackground='#4f4f4f', height=1)
    data_aug_c2_r13.pack(side=TOP, expand=1)




    data_aug_window_bottom = Frame(data_aug_window_bg, bg='#4f4f4f', width=500, height=30)
    data_aug_window_bottom.pack(side=TOP, expand=0, anchor=NW)
    data_aug_window_bottom.pack_propagate(0)
    Augment_button = Button(data_aug_window_bottom, text="Apply", relief=FLAT, overrelief=FLAT, bg='#1f1f1f',
                           fg='#ffffff', activebackground='#4f4f4f', activeforeground='#ffffff', command=Apply_augmentation)
    Augment_button.pack(side=TOP, expand=0, anchor=N)

    data_aug_window_filler = Frame(data_aug_window_bg, bg='#4f4f4f', width=500, height=30)
    data_aug_window_filler.pack(side=TOP, expand=0, anchor=NW)

def Apply_train_par():
    global opt_type, _loss, _metric, epoch_num, b_sz
    opt_type = Optimizer_Name.get()
    loss_name = Loss_Name.get()
    metric_name = Metric_Name.get()
    epoch_num = int(Epochs.get())
    b_sz = int(B_Sz.get())

    if loss_name == "BinaryCrossentropy" or "CategoricalCrossentropy" or "mse" or "mae" or "SparseCategoricalCrossentropy":
        _loss = loss_name
    elif loss_name == "dice_loss":
        _loss = dice_loss
    elif loss_name == "iouloss":
        _loss = iouloss

    if metric_name == "BinaryCrossentropy" or "CategoricalCrossentropy" or "mse":
        _metric = [metric_name]
        return _metric
    if metric_name == "dice_loss":
        _metric = [dice_loss]
        return _metric

    if metric_name == "iouloss":
        _metric = [iouloss]
        return _metric


def Open_train_par_window():
    global Optimizer_Name, Loss_Name, Metric_Name, Epochs, B_Sz

    train_par_window = Toplevel(bg='#4f4f4f')
    train_par_window.title('Define training parameters')
    train_par_window.iconphoto(False, photo)
    train_par_window.geometry("300x300")

    train_par_window_bg = Frame(train_par_window, bg='#4f4f4f', width=400)
    train_par_window_bg.pack(side=LEFT, expand=1, fill=BOTH, anchor=NW, padx=10, pady=10)

    train_par_window_panel = Frame(train_par_window_bg, bg='#4f4f4f', width=380, height=250)
    train_par_window_panel.pack(side=TOP, expand=1, anchor=NW)
    train_par_window_panel.pack_propagate(0)

    train_par_c1_r1 = Frame(train_par_window_panel, bg='#4f4f4f')
    train_par_c1_r1.pack(side=TOP, anchor=W, expand=1)
    train_par_c1_r1_1 = Label(train_par_c1_r1, text="Optimizer:", bg='#4f4f4f', fg='#ffffff', height=1, width=10)
    train_par_c1_r1_1.pack(side=LEFT, anchor=W)
    Optimizer_Name = StringVar()
    Optimizer_Name.set("Adam")
    Optimizer_Type = OptionMenu(train_par_c1_r1, Optimizer_Name, "Optimizer", "Adam", "RMSprop", "SGD", "Adagrad", "Adadelta")
    Optimizer_Type.config(bg="#1f1f1f", fg="#ffffff", bd=0, relief=FLAT, activeforeground='#ffffff',
                      activebackground='#1f1f1f', width=12, highlightthickness=0)
    Optimizer_Type["menu"].config(bg="#1f1f1f", fg="#ffffff", bd=0, activeforeground='#ffffff', activebackground='#4f4f4f')
    Optimizer_Type.pack(side=LEFT, expand=0, fill=X)

    train_par_c1_r2 = Frame(train_par_window_panel, bg='#4f4f4f')
    train_par_c1_r2.pack(side=TOP, anchor=W, expand=1)
    train_par_c1_r2_1 = Label(train_par_c1_r2, text="Loss:", bg='#4f4f4f', fg='#ffffff', height=1, width=10)
    train_par_c1_r2_1.pack(side=LEFT, anchor=W)
    Loss_Name = StringVar()
    Loss_Name.set("BinaryCrossentropy")
    Loss_Type = OptionMenu(train_par_c1_r2, Loss_Name, "Loss", "BinaryCrossentropy",
                           "CategoricalCrossentropy", "mse", "mae", "SparseCategoricalCrossentropy")
    Loss_Type.config(bg="#1f1f1f", fg="#ffffff", bd=0, relief=FLAT, activeforeground='#ffffff',
                          activebackground='#1f1f1f', width=12, highlightthickness=0)
    Loss_Type["menu"].config(bg="#1f1f1f", fg="#ffffff", bd=0, activeforeground='#ffffff',
                                  activebackground='#4f4f4f')
    Loss_Type.pack(side=LEFT, expand=0, fill=X)

    train_par_c1_r3 = Frame(train_par_window_panel, bg='#4f4f4f')
    train_par_c1_r3.pack(side=TOP, anchor=W, expand=1)
    train_par_c1_r3_1 = Label(train_par_c1_r3, text="Metrics", bg='#4f4f4f', fg='#ffffff', height=1, width=10)
    train_par_c1_r3_1.pack(side=LEFT, anchor=W)
    Metric_Name = StringVar()
    Metric_Name.set("mse")
    Metric_Type = OptionMenu(train_par_c1_r3, Metric_Name, "Metric", "BinaryCrossentropy",
                           "CategoricalCrossentropy", "mse", "mae", "SparseCategoricalCrossentropy")
    Metric_Type.config(bg="#1f1f1f", fg="#ffffff", bd=0, relief=FLAT, activeforeground='#ffffff',
                     activebackground='#1f1f1f', width=12, highlightthickness=0)
    Metric_Type["menu"].config(bg="#1f1f1f", fg="#ffffff", bd=0, activeforeground='#ffffff',
                             activebackground='#4f4f4f')
    Metric_Type.pack(side=LEFT, expand=0, fill=X)

    train_par_c1_r4 = Frame(train_par_window_panel, bg='#4f4f4f')
    train_par_c1_r4.pack(side=TOP, anchor=W, expand=1)
    train_par_c1_r4_1 = Label(train_par_c1_r4, text="Epochs:", bg='#4f4f4f', fg='#ffffff', width=10)
    train_par_c1_r4_1.pack(side=LEFT, anchor=W)
    Epochs = Entry(train_par_c1_r4, width=7, bg="#1f1f1f", fg='#ffffff')
    Epochs.insert(0, "200")
    Epochs.pack(side=LEFT, anchor=W)

    train_par_c1_r5 = Frame(train_par_window_panel, bg='#4f4f4f')
    train_par_c1_r5.pack(side=TOP, anchor=W, expand=1)
    train_par_c1_r5_1 = Label(train_par_c1_r5, text="Batch size", bg='#4f4f4f', fg='#ffffff', width=10)
    train_par_c1_r5_1.pack(side=LEFT, anchor=W)
    B_Sz = Entry(train_par_c1_r5, width=7, bg="#1f1f1f", fg='#ffffff')
    B_Sz.insert(0, "1")
    B_Sz.pack(side=LEFT, anchor=W)

    train_par_window_bottom = Frame(train_par_window_bg, bg='#4f4f4f', width=200, height=30)
    train_par_window_bottom.pack(side=TOP, expand=0, anchor=NW)
    train_par_window_bottom.pack_propagate(0)
    train_par_button = Button(train_par_window_bottom, text="Apply", relief=FLAT, overrelief=FLAT, bg='#1f1f1f',
                            fg='#ffffff', activebackground='#4f4f4f', activeforeground='#ffffff',
                            command=Apply_train_par)
    train_par_button.pack(side=TOP, expand=0, anchor=N)

    train_par_window_filler = Frame(train_par_window_bg, bg='#4f4f4f', width=200, height=30)
    train_par_window_filler.pack(side=TOP, expand=0, anchor=NW)


def Load_model_weights():
    # Create list of input images
    Train_input_dir_list = filedialog.askopenfilenames(parent=root, title='Load pre-trained weights')
    current_model.load_weights(Train_input_dir_list[0])



## Training
def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1))

smooth = 10


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_ch1(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true[:,:,:,:1])
    y_pred_f = tf.keras.layers.Flatten()(y_pred[:,:,:,:1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_ch2(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true[:,:,:,1:2])
    y_pred_f = tf.keras.layers.Flatten()(y_pred[:,:,:,1:2])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_ch3(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true[:,:,:,2:])
    y_pred_f = tf.keras.layers.Flatten()(y_pred[:,:,:,2:])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    dice_loss = 1 - dice_coef(y_true, y_pred)
    return dice_loss




def IoU(targets, inputs):
    # flatten label and prediction tensors
    inputs = tf.keras.layers.Flatten()(inputs)
    targets = tf.keras.layers.Flatten()(targets)

    intersection = tf.reduce_sum(targets * inputs)
    total = tf.reduce_sum(targets) + tf.reduce_sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return IoU


def iouloss(targets, inputs):
    iou_loss = 1 - IoU(targets, inputs)
    return iou_loss


def total_loss(targets, inputs):
    dloss = dice_loss(targets, inputs)
    iloss = iouloss(targets, inputs)
    totalloss = iloss + dloss
    return totalloss


def train_model():
    global current_model
    weight_path = Val_target_dir_list[0]
    weight_folder_path = os.path.dirname(weight_path)

    filepath = os.path.join(weight_folder_path, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]


    train_gen = ImageGEN.DataGen(Train_index_list, Train_input_dir_list, Train_target_dir_list, o_w, o_h, t_w, t_h,
                                 rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                                 t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                                 n_p, in_n, tar_n,
                                 b_u, in_br, tar_br, c_u, in_c, tar_c, deep_sup, batch_size=1)
    valid_gen = ImageGEN.DataGen(Val_index_list, Val_input_dir_list, Val_target_dir_list, o_w, o_h, t_w, t_h,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, deep_sup, batch_size=1)

    batch_size = b_sz

    train_steps = len(Train_index_list) // batch_size
    valid_steps = len(Val_index_list) // batch_size

    epochs = epoch_num

    current_model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                      epochs=epochs, callbacks=callbacks_list)


def Save_model_weights():
    User_model_weight_file = filedialog.asksaveasfile(mode='w', defaultextension=".hdf5")
    current_model.save_weights(User_model_weight_file.name)

def Imagesave():
    Save_Folder_Name = filedialog.askdirectory(parent=root, title='Save predicted images under this folder')
    gen = ImageGEN.DataGen(Trans_index_list, Trans_input_dir_list, Trans_input_dir_list, o_w, o_h, t_w, t_h,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, deep_sup, batch_size=1)
    for i in range(0,len(Trans_input_dir_list)):
        Img1, Img2 = gen.__getitem__(i)
        Img3 = current_model.predict(Img1[0:, :, :, :])
        Img3 = Img3[0]
        if deep_sup == 1:
            Img3 = Img3[0:int(t_h), 0:t_w, :]
        Namewithextension = os.path.basename(Trans_input_dir_list[i])
        Namewithoutextension, Extension = os.path.splitext(Namewithextension)
        filename = os.path.join(Save_Folder_Name, Namewithoutextension + 'Predicted file' + '.png')
        print(filename)
        keras.preprocessing.image.save_img(filename, Img3)

############
############

root = Tk()
root.title('FreeAI DeepImageTranslator version 1.0 Copyright (C) 2021 by Run Zhou Ye and En Zhou Ye. All rights reserved.')
photo = PhotoImage(file = "icon.gif")
root.iconphoto(False, photo)
root.geometry("1500x500")


# Create menu
Root_Menu =  Menu(root)
root.config(menu=Root_Menu)

# Create File menu
Root_Menu_File = Menu(Root_Menu, tearoff=FALSE)
Root_Menu.add_cascade(label="File", menu=Root_Menu_File)
Root_Menu_File.add_command(label="Load training images", command=Load_train_input)
Root_Menu_File.add_command(label="Load training targets", command=Load_train_target)
Root_Menu_File.add_command(label="Load validation images", command=Load_val_input)
Root_Menu_File.add_command(label="Load validation targets", command=Load_val_target)

# Add Model menu
Root_Menu_Model = Menu(Root_Menu, tearoff=FALSE)
Root_Menu.add_cascade(label="Model", menu=Root_Menu_Model)
Root_Menu_Model.add_command(label="Set model hyperparameters", command=Open_train_par_window)
Root_Menu_Model.add_command(label="New model", command=Open_new_model_window)
Root_Menu_Model.add_command(label="Load model weights", command=Load_model_weights)
Root_Menu_Model.add_command(label="Save current model weights", command=Save_model_weights)

# Add Train menu
Root_Menu_Train = Menu(Root_Menu, tearoff=FALSE)
Root_Menu.add_cascade(label="Train", menu=Root_Menu_Train)
Root_Menu_Train.add_command(label="Data augmentation", command=Open_data_aug_window)
Root_Menu_Train.add_command(label="Train model", command=train_model)

# Add Translate menu
Root_Menu_Translate = Menu(Root_Menu, tearoff=FALSE)
Root_Menu.add_cascade(label="Translate", menu=Root_Menu_Translate)
Root_Menu_Translate.add_command(label="Load images for translation", command=Load_trans_input)
Root_Menu_Translate.add_command(label="Save translated images", command=Imagesave)


####### Panels:

# Status bar
Status_bar = Label(root, bg="#4f4f4f", height=1, text='Ready       ', fg="white", anchor=E)
Status_bar.pack(fill=BOTH, expand=1, side=BOTTOM)

# Main panel
Main_panel = PanedWindow(root, bd=0, height=1000, bg="#4f4f4f", orient=VERTICAL, sashwidth=20)
Main_panel.pack(fill=BOTH, expand=1, side=TOP)

# Bottom panel
Bottom_panel = Frame(Main_panel, bg="#4f4f4f", height=500)
Main_panel.add(Bottom_panel)

# Image control panel
Control_panel = Frame(Bottom_panel, bg="#4f4f4f", width=100)
Control_panel.pack(side=LEFT, fill=BOTH, expand=1)

image_set = 1
rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r, t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, \
tar_b, n_p, in_n, tar_n, b_u, in_br, tar_br, c_u, in_c, tar_c = \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
current_train_index = 0
current_val_index = 0
current_Trans_index = 0


def forward(event=None):
    global cumulative_train_index, current_train_index, cumulative_val_index, current_val_index, cumulative_Trans_index, current_Trans_index, Img1, Img2, Img3
    if image_set == 1: # Training set selected
        # Update training set index
        gen = ImageGEN.DataGen(Train_index_list, Train_input_dir_list, Train_target_dir_list, o_w, o_h, t_w, t_h,
                               rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                               t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                               n_p, in_n, tar_n,
                               b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
        cumulative_train_index = cumulative_train_index + 1
        current_train_index = (abs(cumulative_train_index)) % (len(Train_index_list))

        Img1, Img2 = gen.__getitem__(current_train_index)

        Img3 = Image.open(
            'iconbw.png')

        try:
            Img3 = current_model.predict(Img1[0:, :, :, :])
            Img3 = Img3[0]
            Img3 = keras.preprocessing.image.array_to_img(Img3)
        except:
            pass

        Img1 = Img1[0]
        Img1 = keras.preprocessing.image.array_to_img(Img1)
        Img2 = Img2[0]
        Img2 = keras.preprocessing.image.array_to_img(Img2)

        Image_panel.add(Image1, stretch="always")
        Image_panel.add(Image2, stretch="always")
        Image_panel.add(Image3, stretch="always")


    if image_set == 2: # Validation set selected
        # Update training set index
        gen = ImageGEN.DataGen(Val_index_list, Val_input_dir_list, Val_target_dir_list, o_w, o_h, t_w, t_h,
                               rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                               t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                               n_p, in_n, tar_n,
                               b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
        cumulative_val_index = cumulative_val_index + 1
        current_val_index = (abs(cumulative_val_index)) % (len(Val_index_list))

        Img1, Img2 = gen.__getitem__(current_val_index)

        Img3 = Image.open(
            'iconbw.png')

        try:
            Img3 = current_model.predict(Img1[0:, :, :, :])
            Img3 = Img3[0]
            Img3 = keras.preprocessing.image.array_to_img(Img3)
        except:
            pass

        Img1 = Img1[0]
        Img1 = keras.preprocessing.image.array_to_img(Img1)
        Img2 = Img2[0]
        Img2 = keras.preprocessing.image.array_to_img(Img2)

        Image_panel.add(Image1, stretch="always")
        Image_panel.add(Image2, stretch="always")
        Image_panel.add(Image3, stretch="always")

    if image_set == 3:  # Translation set selected
        # Update training set index
        gen = ImageGEN.DataGen(Trans_index_list, Trans_input_dir_list, Trans_input_dir_list, o_w, o_h, t_w, t_h,
                               rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                               t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                               n_p, in_n, tar_n,
                               b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
        cumulative_Trans_index = cumulative_Trans_index + 1
        current_Trans_index = (abs(cumulative_Trans_index)) % (len(Trans_index_list))
        Img1, Img2 = gen.__getitem__(current_Trans_index)

        Img3 = Image.open(
            'iconbw.png')

        try:
            Img3 = current_model.predict(Img1[0:, :, :, :])
            Img3 = Img3[0]
            Img3 = keras.preprocessing.image.array_to_img(Img3)
        except:
            pass

        Img1 = Img1[0]
        Img1 = keras.preprocessing.image.array_to_img(Img1)
        Img2 = Img2[0]
        Img2 = keras.preprocessing.image.array_to_img(Img2)

        Image_panel.add(Image1, stretch="always")
        Image_panel.add(Image2, stretch="always")
        Image_panel.add(Image3, stretch="always")


def backward(event=None):
    global cumulative_train_index, current_train_index, cumulative_val_index, current_val_index, cumulative_Trans_index, current_Trans_index, Img1, Img2, Img3
    if image_set == 1: # Training set selected
        # Update training set index
        gen = ImageGEN.DataGen(Train_index_list, Train_input_dir_list, Train_target_dir_list, o_w, o_h, t_w, t_h,
                               rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                               t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                               n_p, in_n, tar_n,
                               b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
        cumulative_train_index = cumulative_train_index - 1
        current_train_index = (abs(cumulative_train_index)) % (len(Train_index_list))

        Img1, Img2 = gen.__getitem__(current_train_index)

        Img3 = Image.open(
            'iconbw.png')

        try:
            Img3 = current_model.predict(Img1[0:, :, :, :])
            Img3 = Img3[0]
            Img3 = keras.preprocessing.image.array_to_img(Img3)
        except:
            pass

        Img1 = Img1[0]
        Img1 = keras.preprocessing.image.array_to_img(Img1)
        Img2 = Img2[0]
        Img2 = keras.preprocessing.image.array_to_img(Img2)

        Image_panel.add(Image1, stretch="always")
        Image_panel.add(Image2, stretch="always")
        Image_panel.add(Image3, stretch="always")


    if image_set == 2: # Validation set selected
        # Update training set index
        gen = ImageGEN.DataGen(Val_index_list, Val_input_dir_list, Val_target_dir_list, o_w, o_h, t_w, t_h,
                               rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                               t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                               n_p, in_n, tar_n,
                               b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
        cumulative_val_index = cumulative_val_index - 1
        current_val_index = (abs(cumulative_val_index)) % (len(Val_index_list))

        Img1, Img2 = gen.__getitem__(current_val_index)

        Img3 = Image.open(
            'iconbw.png')

        try:
            Img3 = current_model.predict(Img1[0:, :, :, :])
            Img3 = Img3[0]
            Img3 = keras.preprocessing.image.array_to_img(Img3)
        except:
            pass

        Img1 = Img1[0]
        Img1 = keras.preprocessing.image.array_to_img(Img1)
        Img2 = Img2[0]
        Img2 = keras.preprocessing.image.array_to_img(Img2)

        Image_panel.add(Image1, stretch="always")
        Image_panel.add(Image2, stretch="always")
        Image_panel.add(Image3, stretch="always")

    if image_set == 3:  # Translation set selected
        # Update training set index
        gen = ImageGEN.DataGen(Trans_index_list, Trans_input_dir_list, Trans_input_dir_list, o_w, o_h, t_w, t_h,
                               rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                               t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                               n_p, in_n, tar_n,
                               b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
        cumulative_Trans_index = cumulative_Trans_index - 1
        current_Trans_index = (abs(cumulative_Trans_index)) % (len(Trans_index_list))
        Img1, Img2 = gen.__getitem__(current_Trans_index)

        Img3 = Image.open(
            'iconbw.png')

        try:
            Img3 = current_model.predict(Img1[0:, :, :, :])
            Img3 = Img3[0]
            Img3 = keras.preprocessing.image.array_to_img(Img3)
        except:
            pass

        Img1 = Img1[0]
        Img1 = keras.preprocessing.image.array_to_img(Img1)
        Img2 = Img2[0]
        Img2 = keras.preprocessing.image.array_to_img(Img2)

        Image_panel.add(Image1, stretch="always")
        Image_panel.add(Image2, stretch="always")
        Image_panel.add(Image3, stretch="always")


def switch_to_train():
    global image_set, Img1, Img2, Img3
    image_set = 1
    gen = ImageGEN.DataGen(Train_index_list, Train_input_dir_list, Train_target_dir_list, o_w, o_h, t_w, t_h,
                           rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                           t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                           n_p, in_n, tar_n,
                           b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
    Img1, Img2 = gen.__getitem__(current_train_index)

    Img3 = Image.open(
        'iconbw.png')

    try:
        Img3 = current_model.predict(Img1[0:, :, :, :])
        Img3 = Img3[0]
        Img3 = keras.preprocessing.image.array_to_img(Img3)
    except:
        pass

    Img1 = Img1[0]
    Img1 = keras.preprocessing.image.array_to_img(Img1)
    Img2 = Img2[0]
    Img2 = keras.preprocessing.image.array_to_img(Img2)

    Image_panel.add(Image1, stretch="always")
    Image_panel.add(Image2, stretch="always")
    Image_panel.add(Image3, stretch="always")


def switch_to_val():
    global image_set, Img1, Img2, Img3
    image_set = 2
    gen = ImageGEN.DataGen(Val_index_list, Val_input_dir_list, Val_target_dir_list, o_w, o_h, t_w, t_h,
                           rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                           t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                           n_p, in_n, tar_n,
                           b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
    Img1, Img2 = gen.__getitem__(current_val_index)

    Img3 = Image.open(
        'iconbw.png')

    try:
        Img3 = current_model.predict(Img1[0:, :, :, :])
        Img3 = Img3[0]
        Img3 = keras.preprocessing.image.array_to_img(Img3)
    except:
        pass

    Img1 = Img1[0]
    Img1 = keras.preprocessing.image.array_to_img(Img1)
    Img2 = Img2[0]
    Img2 = keras.preprocessing.image.array_to_img(Img2)

    Image_panel.add(Image1, stretch="always")
    Image_panel.add(Image2, stretch="always")
    Image_panel.add(Image3, stretch="always")


def switch_to_trans():
    global image_set, Img1, Img2, Img3
    image_set = 3
    gen = ImageGEN.DataGen(Trans_index_list, Trans_input_dir_list, Trans_input_dir_list, o_w, o_h, t_w, t_h,
                           rp_prob, in_rp, tar_rp, s_p, in_s, tar_s, r_d, in_r, tar_r,
                           t_p, in_t, tar_t, in_f, tar_f, d_n, d_s, in_d, tar_d, e_p, in_e, tar_e, b_p, in_b, tar_b,
                           n_p, in_n, tar_n,
                           b_u, in_br, tar_br, c_u, in_c, tar_c, batch_size=1)
    Img1, Img2 = gen.__getitem__(current_Trans_index)
    try:
        Img3 = current_model.predict(Img1[0:, :, :, :])
        Img3 = Img3[0]
        Img3 = keras.preprocessing.image.array_to_img(Img3)
    except:
        pass

    Img1 = Img1[0]
    Img1 = keras.preprocessing.image.array_to_img(Img1)
    Img2 = Img2[0]
    Img2 = keras.preprocessing.image.array_to_img(Img2)

    Image_panel.add(Image1, stretch="always")
    Image_panel.add(Image2, stretch="always")
    Image_panel.add(Image3, stretch="always")



Train_img_button = Button(Control_panel, text="Training", width=4, height=1, bd=0,
                        bg="#4f4f4f", activebackground="#4f4f4f", fg="#8a8a8a", command=switch_to_train)
Val_img_button = Button(Control_panel, text="Validation", width=4, height=1, bd=0,
                        bg="#4f4f4f", activebackground="#4f4f4f", fg="#8a8a8a", command=switch_to_val)
Predicted_img_button = Button(Control_panel, text="Translated", width=4, height=1, bd=0,
                        bg="#4f4f4f", activebackground="#4f4f4f", fg="#8a8a8a", command=switch_to_trans)


Forward_button = Button(Control_panel, text=">", font=('helvetica', 20, 'bold'), width=4, height=1, bd=0,
                        bg="#4f4f4f", activebackground="#4f4f4f", fg="#8a8a8a", command=forward)
Backward_button = Button(Control_panel, text="<", font=('helvetica', 20, 'bold'), width=4, height=1, bd=0,
                        bg="#4f4f4f", activebackground="#4f4f4f", fg="#8a8a8a", command=backward)

Backward_button.pack(side=BOTTOM, fill=BOTH)
Forward_button.pack(side=BOTTOM, fill=BOTH)
Predicted_img_button.pack(side=BOTTOM, fill=BOTH)
Val_img_button.pack(side=BOTTOM, fill=BOTH)
Train_img_button.pack(side=BOTTOM, fill=BOTH)

# Image panels
Image_panel = PanedWindow(Bottom_panel, width=100000, orient=HORIZONTAL, sashwidth=20, bd=0, bg="#4f4f4f")
Image_panel.pack(side=LEFT, fill=BOTH, expand=1)

Image1 = Canvas(Image_panel, bg="#4f4f4f", width=341)
Image_panel.add(Image1, stretch="always")
Img1 = Image.open('iconbw.png')


Image2 = Canvas(Image_panel, bg="#4f4f4f", width=341)
Image_panel.add(Image2, stretch="always")
Img2 = Image.open('iconbw.png')


Image3 = Canvas(Image_panel, bg="#4f4f4f", width=341)
Image_panel.add(Image3, stretch="always")
Img3 = Image.open('iconbw.png')


def resize_Img1(e):
    global resized_Img1, new_Img1
    resized_Img1 = Img1.resize((e.width, e.height), Image.ANTIALIAS)
    new_Img1 = ImageTk.PhotoImage(resized_Img1)
    Image1.create_image(0, 0, image=new_Img1, anchor="nw")
    Image1.create_text(7, 0, text='Original image', fill='red', stipple='gray25', width=e.width, anchor="nw")

def resize_Img2(e):
    global resized_Img2, new_Img2
    resized_Img2 = Img2.resize((e.width, e.height), Image.ANTIALIAS)
    new_Img2 = ImageTk.PhotoImage(resized_Img2)
    Image2.create_image(0, 0, image=new_Img2, anchor="nw")
    Image2.create_text(7, 0, text='Original target', fill='red', stipple='gray25', width=e.width, anchor="nw")

def resize_Img3(e):
    global resized_Img3, new_Img3
    resized_Img3 = Img3.resize((e.width, e.height), Image.ANTIALIAS)
    new_Img3 = ImageTk.PhotoImage(resized_Img3)
    Image3.create_image(0, 0, image=new_Img3, anchor="nw")
    Image3.create_text(7, 0, text='Predicted target', fill='red', stipple='gray25', width=e.width, anchor="nw")

Image1.bind('<Configure>', resize_Img1)
Image2.bind('<Configure>', resize_Img2)
Image3.bind('<Configure>', resize_Img3)
root.bind('<Left>', backward)
root.bind('<Right>', forward)

root.mainloop()
