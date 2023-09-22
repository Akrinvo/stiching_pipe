from tkinter import *
from tkinter import ttk , messagebox
import json
import cv2
from PIL import ImageTk, Image, ImageGrab
from concatt import *
from start import *
import multiprocessing
import threading
from threading import Thread, Lock
from tkinter.messagebox import askyesno
import matplotlib.pyplot as plt
import os
final_img=0
good_img,bad_img = 0,0
dia_lbl = None
# lock = threading.Lock()
# lock =Lock()
image_label8 = None

# label2 =None
# lbl1 = None
# label1 = None

# cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture("assembly2.avi")

def center_window(window):
    window.update_idletasks()
    w = window.winfo_width()
    h = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (w // 2)
    y = (window.winfo_screenheight() // 2) - (h // 2)
    window.geometry(f"{w}x{h}+{x}+{y}")


def preview_camera():
    print("working.......")
    while True:
        imagelist = camera_opner()
        partition=np.ones((imagelist[0].shape[0],20,3),np.uint8)*255
        concat_image=hconcat_resize([imagelist[0],partition,imagelist[1],partition,imagelist[2],partition,imagelist[3]])
        
        cv2.namedWindow("Press-Q for Exit",cv2.WINDOW_NORMAL)

        cv2.imshow("Press-Q for Exit",concat_image)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()  
   


def load_image(path, width, height):
    image = Image.open(path)
    image = image.resize((width,height))  # Resize the image if needed
    image_tk = ImageTk.PhotoImage(image)
    return image_tk



def logo(x):
    heading = Label(x, text="Compaq work", font=("times new roman", 30, "bold"), bg=bg_color,
                fg="black", relief=GROOVE)
    image_tk = load_image("logo.jpeg",150,50)
    image_label2 = Label(heading, image=image_tk)
    image_label2.image = image_tk  # Store a reference to the image to prevent garbage collection
    image_label2.pack(side=LEFT)
    heading.pack(fill=X)

calib_flage = False

#
popup = None

def popup_window():
    popup = Toplevel(root)
    popup.title("INTERFACE")
    popup.overrideredirect(True)
    popup.geometry("400x200")

    label3 = Label(popup, text="working...........",font=("times new roman", 25, "bold"),fg="green")
    label3.pack(padx=20,pady=20) 

    open_another_button2 = Button(popup, text="Exit",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=lambda: popup.destroy())
    open_another_button2.place(x=200, y=150)  

    return popup


def destroy_popup():
    global popup
    if popup:
        popup.destroy()
        print("destroy")



# directory_path(1)

def last_window():
    # global popup
    global calib_flage
    if calib_flage:
        pass
    else:
        calib_flage = True
        last_window = Toplevel(root)
        last_window.title("INTERFACE")
        # last_window.wm_attributes('-fullscreen', 'True')
        last_window.overrideredirect(True)
        last_window.geometry("1500x800")
        center_window(last_window)
        # popup_window()

        logo(last_window)


        def read_options(configuration_root_directory = None):
            filnames = os.listdir(configuration_root_directory)
            dia_list = [filname.split(".")[0][2:]+'.'+filname.split(".")[1] for filname in filnames]
            return dia_list
        
        def label2_work():
            global label2
            label2 = Label(last_window,text="working.....",font=("times new roman", 15, "bold"),padx=10,pady=10)
            label2.pack()

      

        def update_frame1():
        
        
            global final_img,popup
            print("update frame working")
            # popup = popup_window()
            # queue.put("hello")
            # lock.acquire()
            
            dia=clicked.get()
            # ret1, frame1 = cap1.read()
            imagelist=camera_opner()
            partition=np.ones((imagelist[0].shape[0],20,3),np.uint8)*255
            image=hconcat_resize([imagelist[0],partition,imagelist[1],partition,imagelist[2],partition,imagelist[3]])
            # dia=52.66
            with open(f"setting/xy{dia}.json","r") as openfile:
                points=json.load(openfile)
            f_img=final_unwrap(imagelist,two_points=points)
            final_img=calibrated_image(dia,f_img)
            # cv2.imshow("img",final_img)
            # cv2.waitKey(30)
            print("xyz")
            parth=np.ones((20,image.shape[1],3),np.uint8)*255
            frame=vconcat_resize([image,parth,final_img])
            # if ret1:
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame1, (1000, 400))
            img1 = Image.fromarray(frame1)
            img_tk = ImageTk.PhotoImage(image=img1)
            # if label1:
            #     label1.destroy()
            label1.config(image=img_tk)
         
            label1.img1 = img_tk
            # label1.after(10, update_frame1)
            label1.pack()
            # time.sleep(2)
            destroy_popup()
            # popup.destroy()
            # thread1.join()
            print("popup done")
            # queue.put("work done")
            # thread1.join()
            # lock.release()
            # thread1.join()

        def preview_plt():
            global final_img
            img = cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
            plt.figure(num="Image Preview")
            plt.imshow(img)
            plt.show()
                

        def work():
            print("work start")
            update_frame1()
            print("work done")
        # p1 = multiprocessing.Process(target=update_frame1,args=(2,))

        def directory_path(dia):
            print("1")
            base_dir = "SmartViz_Image_Manual_Inspection"
            print("@")
            counter = 1
            dir_nm = f"dia_{dia}_{counter}"
            print("w")

            dir_path = os.path.join(base_dir,dir_nm)
            
            while os.path.exists(dir_path):
                print("#")
                counter += 1
                dir_nm = f"dia_{dia}_{counter}"
                print("$")
                dir_path = os.path.join(base_dir,dir_nm)


            os.mkdir(dir_path) 
            print("d")   
            return dir_path

            
        # thread1 = threading.Thread(target=update_frame1,args=(10,lock))
        def selected(event):
            global popup, dia
            
            # thread1 = threading.Thread(target=update_frame1)
            dia = clicked.get()
            dia_lbl.config(text=f"dia_{dia}")
            dia_lbl.pack()
            print("123")

            
            # print("work before....")
            print("popup start")

            popup = popup_window()
            p2 = multiprocessing.Process(target=update_frame1)
            # directory()
            p2.run()
            print("update completed")
            # directory_path(dia)
            

            
            # update_frame1()
            # thread1.start()
            # queue = multiprocessing.Queue()
            # print("processing start...")

            
          
            
            # p1.join()
            # update_frame1()
            
            print("done....")
            # update_frame1()
            print("process")
            # new_w.destroy()
            
                
     
        

        def save_good():
            global final_img,good_img,dia
            print(dia)
            dir_path = directory_path(dia)
            
            print("123")
            good_img += 1
            print(good_img)
            good_dir=  os.path.join(dir_path,"OK")  
            # last = os.path.join(good_dir,final_img)  
            isExist = os.path.exists(good_dir)
            print(isExist)
            if not isExist:
                os.makedirs(good_dir)
            
            cv2.imwrite(f"{good_dir}/{good_img}.jpg",final_img) 

        def save_bad():
            global final_img,bad_img,dia
            dir_path = directory_path(dia)
            bad_dir = os.path.join(dir_path,"NG")
            bad_img += 1
            # filname="bad/"    
            isExist = os.path.exists(bad_dir)
            if not isExist:
                os.makedirs(bad_dir)
            cv2.imwrite(f"{bad_dir}/{bad_img}.jpg",final_img)


        options = read_options(configuration_root_directory="setting")
        clicked = StringVar()
        clicked.set(options[0])

        drop = OptionMenu(last_window, clicked, *options, command=selected)

        drop.pack(pady=5)

        diameter = LabelFrame(last_window,text="Serial No : ",font=("times new roman", 15, "bold"))
        dia_lbl = Label(diameter,text=f"dia",font=("times new roman", 10, "bold"), padx=10, pady=10)
        # dia_lbl.pack()
        diameter.place(x=80, y=70)

        lbl1 = LabelFrame(last_window, text="Results",font=("times new roman", 15, "bold"), padx=10, pady=10)
        # lbl1.pack(side=LEFT)
        lbl1.place(x=200, y=150)
        label1 = Label(lbl1)
        # label1.pack()

        def destroy_last_window():
            global calib_flage
            calib_flage = False
            last_window.destroy()

        def on_close():
            global calib_flage
            calib_flage = False
            print(calib_flage)
            last_window.destroy()

        open_another_button = Button(last_window, text="Next",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=work)
        open_another_button.place(x=500,y=650)

        open_another_button = Button(last_window, text="Preview",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=preview_plt)
        open_another_button.place(x=800,y=650)

        open_another_button1 = Button(last_window, text="Exit",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=destroy_last_window)
        open_another_button1.place(x=1320,y=750)

        open_another_button2 = Button(last_window, text="OK",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=lambda:save_good())
        open_another_button2.place(x=10, y=750)
        
        open_another_button3 = Button(last_window, text="NG",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=lambda:save_bad())
        open_another_button3.place(x=200, y=750)

        last_window.protocol("Another Window",on_close)
    

def open_new_window():
    
    global calib_flage
    if calib_flage:
        pass
    else:   
        calib_flage = True
        global input_value
        new_window = Toplevel(root)
        new_window.title("CALIBRATION")
        new_window.overrideredirect(True)
        new_window.geometry('1800x700')
        center_window(new_window)
        
        try: os.remove("output.jpg")
        except:pass
        
        logo(new_window)

        main_frame = LabelFrame(new_window, text="For Selection Of Images", font=("times new roman", 30, "bold"),fg="black", relief=GROOVE)
        text1 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
        mini_text = Label(text1, text="Image 1 - Press 1 ", font=("times new roman", 15, "bold"))
        mini_text.grid(row=0,column=0,padx=5,pady=5)
        text1.grid(row=0,column=0,padx=10, pady=10)
        text2 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
        mini_text = Label(text2,text="Image 2 - Press 2 ", font=("times new roman", 15, "bold"))
        mini_text.grid(padx=5,pady=5)
        text2.grid(row=0, column=1, padx=10,pady=10)
        text3 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
        mini_text = Label(text3,text="Image 3 - Press 3 ", font=("times new roman", 15, "bold"))
        mini_text.grid(padx=5,pady=5)
        text3.grid(row=0, column=2, padx=10,pady=10)
        text4 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
        mini_text = Label(text4,text="Image 4 - Press 4 ", font=("times new roman", 15, "bold"))
        mini_text.grid(padx=5,pady=5)
        text4.grid(row=0, column=3, padx=10,pady=10)
        text5 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
        mini_text = Label(text5,text="a -> Left ", font=("times new roman", 15, "bold"))
        mini_text.grid(row=0, column=1)
        mini_text = Label(text5,text="s -> Down ", font=("times new roman", 15, "bold"))
        mini_text.grid(row=1, column=0)
        mini_text = Label(text5,text="w -> Up ", font=("times new roman", 15, "bold"))
        mini_text.grid(row=0,column=0 )
        mini_text = Label(text5,text="d -> Right ", font=("times new roman", 15, "bold"))
        mini_text.grid(row=1,column=1)
        text5.grid(row=0, column=4, padx=10,pady=10)
        text6 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
        mini_text = Label(text6,text="Press F after Calibration is done", font=("times new roman", 15, "bold"))
        mini_text.grid(padx=5,pady=5)
        text6.grid(row=0, column=5, padx=10,pady=10)
        main_frame.pack(pady=10,padx=15)

        input_label = Label(new_window, text="Diameter:",font=("times new roman",20, "bold"),fg="white", bg="grey", height=1, width=10)
        input_label.pack(pady=10)
        input_entry = Entry(new_window)
        input_entry.pack()
        lbl1 = LabelFrame(new_window, text="work",font=("times new roman", 15, "bold"), padx=10, pady=10)
        # label1.pack()
        # lbl1.pack(side=LEFT)
        def confirm():
            print("xyz...........")
            ans = askyesno(title="exit",message="do you want to exit")
            if ans:
                new_window.destroy()

        def img_show():
            global image_label8
            image = Image.open("output.jpg")
            image = image.resize((1000,300))  
            image_tk = ImageTk.PhotoImage(image)
            if image_label8:
                image_label8.destroy()
            # image_tk8 = load_image("images/spring.jpg",130,130)
            image_label8 = Label(lbl1, image=image_tk)
            image_label8.image = image_tk
            image_label8.pack()
            lbl1.pack()

        def pass_input_value():
            global input_value, dropdown_values
           
            input_value = input_entry.get()
            print("Input Value:", input_value)
            x1,y2=predict_points(float(input_value))
            points=[[x1,  0.15525],[ 0.5,y2]]
            cor=json.dumps(points)
            with open(f"setting/xy{input_value}.json", "w") as dic:
                        dic.write(cor)
            imagelist=camera_opner()
            
            f_img=final_unwrap(imagelist,two_points=points)
            
            
            
            mover_image(f_img,input_value)
            img_show()

        def destroy_fun():
            global calib_flage
            calib_flage = False
            new_window.destroy()
            
        
        open_another_button = Button(new_window, text="Calibrate",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=20, command=pass_input_value)
        open_another_button.pack(pady=20)

        
        
        exit_button = Button(new_window,text='Exit',font=("times new roman", 20, "bold"),command=destroy_fun,bg="red",fg="white")
        exit_button.place(x=1700,y=655)
        new_window.protocol("NW_Delete_Window",confirm)



root = Tk()
root.title("INTERFACE")
root.geometry("700x500")
root.overrideredirect(True)
center_window(root )
bg_color = "white"

logo(root)

# update_video()
# main_frame = Label(root, text="", font=("times new roman", 30, "bold"),fg="white", relief=GROOVE)
new_window_button = Button(root, text="Calibrate",font=("times new roman",20, "bold"), command=open_new_window,fg="white", bg="Blue", height=1, width=10)
new_window_button1 = Button(root,text="Start",font=("times new roman", 20, "bold"),command=last_window,width=10, height=1, fg="white", bg="green")
# dropdown_button = Button(root, text="Select", command=show_dropdown)
exit_button = Button(root,text='Exit',font=("times new roman", 20, "bold"),command=lambda: root.quit(),bg="red",fg="white")


new_window_button.pack(expand=True )
new_window_button1.pack(expand=True)
exit_button.pack(padx=5, pady=5, side=RIGHT)

new_window_button2 = Button(root, text="Preview",font=("times new roman",15, "bold"), command=preview_camera,fg="white", bg="Blue", height=1, width=10)
new_window_button2.pack(side=LEFT)

root.mainloop()