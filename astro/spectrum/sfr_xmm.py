#!/usr/bin/env python

import sys
import os
import exceptions
import re

grppha_comm="aa"


def extract_mos(regdes,table_name,temp_pi,bg_table_name,out_bg_name,rmf_name,arf_name,spec_bin_cmd,det='mos1'):
    expr="'((X,Y) IN %s)'"%regdes
    out_pi_name = "_a.pi"
    if not re.search(".pi$",bg_table_name):
        cmd="evselect"
        cmd+=" table=%s withspectrumset=yes spectrumset=%s energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=11999 expression=%s "%(bg_table_name,out_bg_name,expr)
        print(cmd)
        cmd+=">/dev/null"
        print("extracting background...")
        os.system(cmd)
    else:
        out_bg_name=bg_table_name

    cmd="evselect"
    cmd+=" table=%s withspectrumset=yes spectrumset=%s energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=11999 expression=%s"%(table_name,out_pi_name,expr)    
    print(cmd)
    cmd+=">/dev/null"
    print("extracting spectrum...")
    os.system(cmd)
 

    try:
        os.unlink(temp_pi)
    except exceptions.OSError:
        pass
    
    cmd="grppha infile='%s' outfile='%s' comm='%s & chkey BACKFILE %s & chkey RESPFILE %s & chkey ANCRFILE %s & exit &'"%(out_pi_name,temp_pi,spec_bin_cmd,out_bg_name,rmf_name,arf_name)
#    print cmd
    cmd+=">/dev/null"
    os.system(cmd)

   

def extract_pn(regdes,table_name,temp_pi,bg_table_name,out_bg_name,rmf_name,arf_name,spec_bin_cmd,det='mos1'):
    expr="'((X,Y) IN %s)'"%regdes
    out_pi_name = "_a.pi"
    if not re.search(".pi$",bg_table_name):
        cmd="evselect"
        cmd+=" table=%s withspectrumset=yes spectrumset=%s energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=20479 expression=%s "%(bg_table_name,out_bg_name,expr)
        print(cmd)
        cmd+=">/dev/null"
        print("extracting background...")
        os.system(cmd)
    else:
        out_bg_name=bg_table_name

    cmd="evselect"
    cmd+=" table=%s withspectrumset=yes spectrumset=%s energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=20479 expression=%s"%(table_name,out_pi_name,expr)    
    print(cmd)
    cmd+=">/dev/null"
    print("extracting spectrum...")
    os.system(cmd)

   
    try:
        os.unlink(temp_pi)
    except exceptions.OSError:
        pass
    
    cmd="grppha infile='%s' outfile='%s' comm='%s & chkey BACKFILE %s & chkey RESPFILE %s & chkey ANCRFILE %s & exit &'"%(out_pi_name,temp_pi,spec_bin_cmd,out_bg_name,rmf_name,arf_name)
#    print cmd
    cmd+=">/dev/null"
    os.system(cmd)



def make_fit_script(fit_script_template,pi_list,fit_script):
    fit_script.write("set ff [open %s w]\n"%temp_result_name)
#    print "fit_scriptfit_script"
    for i in fit_script_template:
        if re.search("^#PARA",i):
            para=re.split("\ +",i)
            fit_script.write("tclout para %s\n"%para[2])
            fit_script.write('puts $ff "$xspec_tclout %s"\n'%para[1])
        elif re.search("^#STAT",i):
            fit_script.write("tclout stat\n")
            fit_script.write('puts $ff "$xspec_tclout statistic"\n')
            fit_script.write("tclout dof\n")
            fit_script.write('puts $ff "$xspec_tclout dof"\n')
        elif re.search("^#GRPPHA",i):
            grppha_comm=str.strip(re.sub("#GRPPHA","",i))
            return grppha_comm
        elif re.search("^#MODEL",i):
            fit_script.write(str.strip(re.sub("#MODEL","",i))+"\n")
        elif re.search("^#DATA",i):
            for j in range(0,len(pi_list)):
                fit_script.write("data %d:%d %s\n"%(j+1,j+1,pi_list[j]))
        elif re.search("^#IGNORE",i):
            for j in range(0,len(pi_list)):
                fit_script.write("ignore %d:%s\n"%(j+1,str.strip(re.sub("#IGNORE","",i))))
        else:
            fit_script.write(i)


    fit_script.write("tclexit\n")
#    print "fdsf"

    
def parse_result(result_file):
    result=""
    for i in result_file:
        str.strip(i)
        sl=re.split("\ +",i)
        
        if re.search('statistic',i):
            result+=sl[0].strip()
            result+=" "
        elif re.search('dof',i):
            result+=sl[0].strip()
            result+=" "
        else:
            result+=sl[0].strip()
            result+=" "
            result+=sl[6].strip()
            result+=" "
    return result




temp_mos1_pi="_mos1.pi"
temp_mos2_pi="_mos2.pi"
temp_pn_pi="_pn.pi"
temp_mos1_bg="_mos1.bg"
temp_mos2_bg="_mos2.bg"
temp_pn_bg="_pn.bg"
temp_fit_script="_a.tcl"
temp_result_name="_a.dat"


def main(argv):
    if len(argv)!=15:
        print("Usage:%s <sample list> <mos1.evt> <mos2.evt> <pn.evt> <mos1.bkg> <mos2.bkg> <pn.bkg> <output file> <mos1.arf> <mos2.arf> <pn.arf> <mos1.rmf> <mos2.rmf> <pn.rmf> <fit script>"%(sys.argv[0]))
        sys.exit()


    sample_list_name=argv[0]
    mos1_name=argv[1]
    mos2_name=argv[2]
    pn_name=argv[3]
    mos1_bkg=argv[4]
    mos2_bkg=argv[5]
    pn_bkg=argv[6]
    output_name=argv[7]
    mos1_arf=argv[8]
    mos2_arf=argv[9]
    pn_arf=argv[10]
    mos1_rmf=argv[11]
    mos2_rmf=argv[12]
    pn_rmf=argv[13]
    
    fit_script_template=argv[14]
    
#    print fit_script_template
    
    
    n=1
    fit_script=open(temp_fit_script,'w');
    grppha_comm=make_fit_script(open(fit_script_template),[temp_mos1_pi,temp_mos2_pi,temp_pn_pi],fit_script)
    #fit_script.write("tclexit")
    
    
    fit_script.write("tclexit\n")
    fit_script.close()
#    print grppha_comm
#    sys.exit()
    outfile=open(output_name,'w')
    for i in open(sample_list_name):
        regdesc=str.strip(i)
        if not re.search('^#NULL',regdesc):
            extract_mos(regdesc,mos1_name,temp_mos1_pi,mos1_bkg,temp_mos1_bg,mos1_rmf,mos1_arf,grppha_comm)
            extract_mos(regdesc,mos2_name,temp_mos2_pi,mos2_bkg,temp_mos2_bg,mos2_rmf,mos2_arf,grppha_comm)
            extract_pn(regdesc,pn_name,temp_pn_pi,pn_bkg,temp_pn_bg,pn_rmf,pn_arf,grppha_comm)
            print("fitting...")
            os.system("xspec %s >/dev/null"%temp_fit_script)
            result=parse_result(open(temp_result_name))
            outfile.write("%d %s %s"%(n,result,'\n'))
            print("%d %s"%(n,result))
        else:
            outfile.write("\n")
            print("%d"%(n))
        n+=1
        outfile.flush()

        
if __name__=='__main__':
    main(sys.argv[1:])
