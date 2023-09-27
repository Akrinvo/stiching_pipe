from tkinter import *
from tkinter import ttk , messagebox
import json
import re
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
pre_v=0
Imagelist=None
serial = False
image_label8 = None
dia = 0
path = None
next_b= False
label2 =None
# lbl1 = None
label3 = None
label4 = None
x = 0
label_prev=None
image_index=0
next_v=0
capture_w =False

calib_flage = False

def preview_camera():
    global calib_flage
    
    if calib_flage:
        pass
    else:
        calib_flage = True
        print("working.......")
        while True:
            imagelist = camera_opner()
            partition=np.ones((imagelist[0].shape[0],20,3),np.uint8)*255
            concat_image=hconcat_resize([imagelist[0],partition,imagelist[1],partition,imagelist[2],partition,imagelist[3]])
            
            cv2.namedWindow("Press-Q for Exit",cv2.WINDOW_NORMAL)

            cv2.imshow("Press-Q for Exit",concat_image)
            if cv2.waitKey(1) == ord("q"):
                calib_flage = False
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

def save_good():
        global final_img,good_img,path, capture_w
        # good_img = 0
        if capture_w:
            # good_img += 1
            good_dir= path
            # print(good_dir)
            # print(len(os.listdir(good_dir)))
            good_img = len(os.listdir(good_dir))
            good_img += 1
            
            # dir_nm = "good"
            # dir_path = os.path.join(good_dir,dir_nm)
            # if not os.path.exists(dir_path):
            #     os.mkdir(dir_path)
            d_path = f"{good_dir}/{good_img}.jpg"
            # if not os.path.exists(d_path):
            #     cv2.imwrite(f"{good_dir}/{good_img}.jpg",final_img) 
            while os.path.exists(d_path):
            #     print("#")
                good_img += 1
                d_path = f"{good_dir}/{good_img}.jpg"
                print("$")
            print(good_img)
            cv2.imwrite(d_path,final_img) 
            messagebox.showinfo("showinfo","saved")

        else:
            messagebox.showerror("showerror","Please Capture the image first..")

def save_bad():
    global final_img,bad_img,path,good_img,capture_w
    # good_img = 0
    if capture_w:
        # good_img += 1
        filname=path  
        good_img = len(os.listdir(filname))
        good_img += 1
        # dir_nm = "bad"
        # dir_path = os.path.join(filname,dir_nm)
        # if not os.path.exists(dir_path):
        #     os.mkdir(dir_path) 
        d_path = f"{filname}/ng_{good_img}.jpg"
        while os.path.exists(d_path):
        #     print("#")
            good_img += 1
            d_path = f"{filname}/ng_{good_img}.jpg"
            print("$")
        print(good_img)
        cv2.imwrite(d_path,final_img)
        messagebox.showinfo("showinfo","saved")

    else:
        messagebox.showerror("showerror","Please Capture the image first..")

# directory_path(1)

def last_window():
    # global popup
    global calib_flage,label_prev
    def exit_function():
        global calib_flage
        messagebox.showwarning("showwarning","Do u want to exit?")
        last_window.destroy()
        calib_flage = False
    if calib_flage:
        pass
    else:
        calib_flage = True
        last_window = Toplevel(root)
        last_window.title("INTERFACE")
        # last_window.wm_attributes('-fullscreen', 'True')
        # last_window.overrideredirect(True)
        last_window.geometry("1500x800")
        last_window.protocol("WM_DELETE_WINDOW",exit_function)
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

        def wait():
             open_another_button.config(text="please wait....")
             print("hellio")
        # thread1 = threading.Thread(target=update_frame1,args=(10,lock))
        
           

        def update_frame1():
        
            # thread1.join()
        
            global popup,Imagelist
            print("update frame working")
            
            # popup = popup_window()
            # queue.put("hello")
            # lock.acquire()
            
            dia=clicked.get()
            # ret1, frame1 = cap1.read()
            Imagelist=camera_opner()
            partition=np.ones((Imagelist[0].shape[0],20,3),np.uint8)*255
            image=hconcat_resize([Imagelist[0],partition,Imagelist[1],partition,Imagelist[2],partition,Imagelist[3]])
            # dia=52.66
            
            # cv2.imshow("img",final_img)
            # cv2.waitKey(30)
            # print("xyz")
            # parth=np.ones((20,image.shape[1],3),np.uint8)*255
            # frame=vconcat_resize([image,parth,final_img])
            # cv2.imshow("img",frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # if ret1:
            frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame1, (1000, 200))
            img1 = Image.fromarray(frame1)
            img_tk = ImageTk.PhotoImage(image=img1)
            label1.config(image=img_tk)
         
            label1.img1 = img_tk
            # label1.after(10, update_frame1)
            label1.pack()



            # time.sleep(2)
            # destroy_popup()
            # popup.destroy()
            # thread1.join()
            print("popup done")
            # queue.put("work done")
            # thread1.join()
            # lock.release()
            # thread1.join()
        
        def preview_plt():
            global final_img,serial,dia,capture_w
            if dia == 0:
                messagebox.showwarning("showwarning", "Please select the diameter...")
            elif len(input_entry.get()) == 0:
                messagebox.showwarning("showwarning", "Please fill the serial no...")
            elif not capture_w:
                messagebox.showerror("showerror","Please capture the image first..")
            else:
                
                img = cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
                plt.figure(num="Image Preview")
                plt.imshow(img)
                plt.show()
            # else:
            #     messagebox.showwarning("showwarning", "Please fill the serial no and select the diameter...")
            
                

        def work():
            print("work start")
            global dia, label2,capture_w
            if label2:
                label2.destroy()
            if dia == 0:
                messagebox.showwarning("showwarning", "Please select the diameter...")
            elif len(input_entry.get()) == 0:
                messagebox.showwarning("showwarning", "Please fill the serial no...")
            elif not serial:
                messagebox.showerror("showerror","Please save the serial no..")
            else:
                capture_w = False
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
        
        def directory_path1(dia,x):
            print("1")
            base_dir = "SmartViz_Image_Manual_Inspection"
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            print("@")
            # counter = 1
            dir_nm = f"dia_{dia}_{x}"
            print("w")

            dir_path = os.path.join(base_dir,dir_nm)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            
            # while os.path.exists(dir_path):
            #     print("#")
            #     counter += 1
            #     dir_nm = f"dia_{dia}_{counter}"
            #     print("$")
            #     dir_path = os.path.join(base_dir,dir_nm)


            # os.mkdir(dir_path) 
            print("d")   
            return dir_path

        def wait():
             open_another_button.config(text="please wait....")
             print("hellio")
        # thread1 = threading.Thread(target=update_frame1,args=(10,lock))



        def selected(event):
            global popup, dia,x, label2,serial
            serial = False
            if label2:
                label2.destroy()
           

            
            # thread1 = threading.Thread(target=update_frame1)
            dia = clicked.get()
            dia_lbl.config(text=f"dia_{dia}")
            # dia_lbl.pack()
            print("123")

            
            # print("work before....")
            print("popup start")

            # popup = popup_window()
            # p2 = multiprocessing.Process(target=update_frame1)
            # directory()
            # p2.run()
            print("update completed")
            # directory_path(dia)
            

            
            update_frame1()
            # thread1.start()
            # queue = multiprocessing.Queue()
            # print("processing start...")

            
          
            
            # p1.join()
            # update_frame1()
            
            print("done....")
            # update_frame1()
            print("process")
            # new_w.destroy()
            
        
        def serial_no():
            global dia,serial,path
            # serial = True
            if dia == 0:
                messagebox.showwarning("showwarning","Please select the diameter..")
            elif len(input_entry.get()) == 0:
                # print("xx")
                messagebox.showwarning("showwarning", "Please fill the serial no ...")
            # elif serial and dia != 0:
            #     messagebox.showwarning("showwarning", "Please Save...")
            else:   
                x = input_entry.get()
                # dia = clicked.get()
                # if dia_lbl:
                #     dia_lbl.destroy()
                serial = True
                dia_lbl.config(text=f"dia_{dia}_{x}")
                path = directory_path1(dia,x)
                # messagebox.showinfo("showinfo","Saved")
                # dia_lbl.pack()
        
        def sort_1(item):
            match = re.search(r'\d+',item)
            if match:
                return int(match.group())
            else:
                return item
       
       
        def next_image():
            global path,good_img,serial,label3,label2,next_v,label_prev,image_index,pre_v

            if dia == 0:
                messagebox.showwarning("showwarning", "Please select the diameter...")
            elif len(input_entry.get()) == 0:
                messagebox.showwarning("showwarning", "Please fill the serial no...")
            elif not serial:
                messagebox.showerror("showerror","Please save the serial no..")
            else:
                folder_path = path

                all_files = os.listdir(folder_path)
                
                all_files = sorted(all_files, key=sort_1)
                x = len(all_files)
                if x == 0:
                    messagebox.showinfo("showinfo","Empty folder")
                elif x==1:
                    messagebox.showinfo("showinfo","only one image exist")

                    
                else:

 
                    x = len(all_files)
                    if x > 0 :
                        
                        image_index=image_index+1
                        print(f"xxxxx  {image_index}  xxxxxxx")

                        if image_index<len(all_files):

                          
                            
                            last_file = os.path.join(folder_path,all_files[image_index])
                            # print(last_file)
                            img = cv2.imread(last_file)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            plt.figure(num="Image Preview")
                            plt.title(f'{all_files[image_index]}')
                            plt.imshow(img)
                        else:
                            image_index=len(all_files)-1
                           
                            messagebox.showinfo("showinfo","end")
                        plt.show()
            
            
            pre_v-=1
            next_v+=1
            if pre_v<0:
                pre_v=0
            if next_v>len(all_files)-1:
                next_v=len(all_files)-1

            

        def previous():
            global path,good_img,serial,label3,label2,pre_v,label_prev,image_index
            if dia == 0:
                messagebox.showwarning("showwarning", "Please select the diameter...")
            elif len(input_entry.get()) == 0:
                messagebox.showwarning("showwarning", "Please fill the serial no...")
            elif not serial:
                messagebox.showerror("showerror","Please save the serial no..")
            else:
                folder_path = path
                

                all_files = os.listdir(folder_path)
                
                all_files = sorted(all_files, key=sort_1)
                x = len(all_files)
                if x == 0:
                    messagebox.showinfo("showinfo","Empty folder")
                elif x==1:
                    last_file = os.path.join(folder_path,all_files[0])
                    img = cv2.imread(last_file)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    plt.figure(num="Image Preview")
                    plt.title(f'{all_files[0]}')

                    plt.imshow(img)
                    
                    
                else:
                
                    x = len(all_files)
                    if x > 0 :
                        x -= 1+pre_v
                        if x>=0:  image_index=x
                        print(x)
                        if pre_v!=None:
                            
                           
                            last_file = os.path.join(folder_path,all_files[image_index])
                            # print(last_file)
                            img = cv2.imread(last_file)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            plt.figure(num="Image Preview")
                            plt.title(f'{all_files[image_index]}')
                            plt.imshow(img)
                    
                           
                            
                            plt.show()
            if pre_v<len(all_files)-1:
                pre_v+=1
            else:messagebox.showinfo("showinfo","end")
            

                        

                    # cv2.imshow("previous",img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

        def capture():
            global dia,Imagelist,final_img,serial,capture_w,label2,next_b,label3,label4
            # if serial:
            #     with open(f"setting/xy{dia}.json","r") as openfile:
            #         points=json.load(openfile)
            #     f_img=final_unwrap(Imagelist,two_points=points)
            #     final_img=calibrated_image(dia,f_img)
            #     frame2 = cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
            #     frame2 = cv2.resize(frame2,(1000,200))
            #     img2 = Image.fromarray(frame2)
            #     img_tk1 = ImageTk.PhotoImage(image=img2)


            #     label2.config(image=img_tk1)
            #     label2.img2 = img_tk1
            #     label2.pack()
            if dia == 0:
                messagebox.showwarning("showwarning", "Please select the diameter...")
            elif len(input_entry.get()) == 0:
                messagebox.showwarning("showwarning", "Please fill the serial no...")
            elif not serial:
                messagebox.showerror("showerror","Please save the serial no..")
            
            else:
                next_b = True
                with open(f"setting/xy{dia}.json","r") as openfile:
                    points=json.load(openfile)
                f_img=final_unwrap(Imagelist,two_points=points)
                final_img=calibrated_image(dia,f_img)
                frame2 = cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
                frame2 = cv2.resize(frame2,(1000,200))
                img2 = Image.fromarray(frame2)
                img_tk1 = ImageTk.PhotoImage(image=img2)
                capture_w = True
                if label3:
                    label3.destroy()
                # if label4:
                #     label4.destroy()    
                if label2:
                    label2.destroy()
                label2 = Label(lbl1,image=img_tk1)
                # label2.config(image=img_tk1)
                label2.image = img_tk1
                label2.pack()
            
       

        options = read_options(configuration_root_directory="setting")
        clicked = StringVar()
        clicked.set("select dia")

        drop = OptionMenu(last_window, clicked, *options, command=selected)

        drop.pack(pady=5)

        diameter = LabelFrame(last_window,text="Serial No : ",font=("times new roman", 15, "bold"))
        dia_lbl = Label(diameter,text=f"dia",font=("times new roman", 10, "bold"), padx=10, pady=10)
        dia_lbl.pack()
        diameter.place(x=10, y=70)

        canvas = Canvas(last_window,height=700,width=1200 )
    
        canvas.pack()

        canvas.create_rectangle(30, 50, 1100, 500,outline="#fb0")

        lbl1 = LabelFrame(last_window, text="",font=("times new roman", 15, "bold"), padx=10, pady=10)
        # lbl1.pack(side=LEFT)
        lbl1.place(x=200, y=150)
        label1 = Label(lbl1)
        label2 = Label(lbl1)
        lbl2 = LabelFrame(last_window,text=" ",font=("times new roman", 15, "bold"), padx=10, pady=10)
        label3 = Label(lbl1)
        label_prev=Label(lbl2,text=" ")
        lbl2.place(x=40,y=450)
        # label_prev.pack()
        # lbl2.place(x=200,y=350)
        # label1.pack()
        

        # label5 = Label(last_window,text="Previous Image",font=("times new roman", 10, "bold"), padx=10, pady=10)
        # label5.place(x=20,y=200)

        def destroy_last_window():
            global calib_flage,dia,serial
            dia = 0
            serial = False
            calib_flage = False
            last_window.destroy()

        def on_close():
            global calib_flage
            calib_flage = False
            print(calib_flage)
            last_window.destroy()

        input_label = Label(last_window, text="Serial no :",font=("times new roman",10, "bold"),fg="black", height=1, width=10)
        input_label.place(x=1230, y=60)
        input_entry = Entry(last_window)
        input_entry.place(x=1200,y=80)

        
        open_another_button = Button(last_window, text="save",font=("times new roman",8, "bold"),fg="white", bg="green", height=1, width=5,command=serial_no)
        open_another_button.place(x=1240,y=100)


        x = input_entry.get()
        # update_frame1()
        # print(x)   

        open_another_button = Button(last_window, text="Forward",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=work)
        open_another_button.place(x=170,y=650)

        open_another_button4 = Button(last_window, text="Preview",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=preview_plt)
        open_another_button4.place(x=640,y=650)

        open_another_button5 = Button(last_window, text="Previous",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=previous)
        open_another_button5.place(x=950,y=650)

        open_another_button6 = Button(last_window, text="Capture",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=capture)
        open_another_button6.place(x=700,y=750)

        open_another_button7 = Button(last_window, text="Next",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=next_image)
        open_another_button7.place(x=1100,y=650)

        open_another_button8 = Button(last_window, text="Backward",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=capture)
        open_another_button8.place(x=320,y=650)

        open_another_button1 = Button(last_window, text="Exit",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=destroy_last_window)
        open_another_button1.place(x=1320,y=750)

        open_another_button2 = Button(last_window, text="OK",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=lambda:save_good())
        open_another_button2.place(x=10, y=750)
        
        open_another_button3 = Button(last_window, text="NG",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=lambda:save_bad())
        open_another_button3.place(x=180, y=750)

        last_window.protocol("Another Window",on_close)
    

def open_new_window():
    
    global calib_flage
    def exit_function1():
        global calib_flage
        messagebox.showwarning("showwarning","Do u want to exit?")
        new_window.destroy()
        calib_flage = False
    if calib_flage:
        pass
    else:   
        calib_flage = True
        global input_value
        new_window = Toplevel(root)
        new_window.title("CALIBRATION")
        # new_window.overrideredirect(True)
        new_window.geometry('2000x700')
        new_window.protocol("WM_DELETE_WINDOW",exit_function1)
        
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
        exit_button.place(x=1765,y=655)
        new_window.protocol("NW_Delete_Window",confirm)



root = Tk()
root.title("INTERFACE")
root.geometry("700x500")
# root.overrideredirect(True)
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