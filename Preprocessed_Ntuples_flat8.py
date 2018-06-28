import ROOT
from ROOT import TH1F, TCanvas, TFile, TColor,THStack,TLegend,TLatex, TChain, TVector3
import numpy as np




FLEt = open("PreLEtTau_8_ViEt15Et0Eta1.4_NNTau_Train.data","w")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

signalFile = TFile("v2/output_Z200.root")
signal = signalFile.Get("mytree") 
backFile = TFile("v2/output_MB200.root")
back = backFile.Get("mytree")

class array(list):
	def swap1D(self, i, j):
		self[i], self[j] = self[j], self[i]
	def swap2D(self, i, j, n):
		for k in range(n):
			self.swap1D(i-k,j+k)
def Preprocess(L1D_raw):
        L1D = np.asarray(L1D_raw)
	L2D = L1D.reshape((17,5)).transpose()
        #Begin real preprocess
        
	phi0 = L2D[1,:]
	phi2 = L2D[3,:]
	if sum(phi0) < sum(phi2):
		L2D = array(list(L2D))
		L2D.swap1D(1,3)
		L2D = np.asarray(L2D)
	right = L2D[1:4,9:14]
	left = L2D[1:4,3:8]
	if sum(sum(right)) < sum(sum(left)):
		L2D = L2D.transpose()
		L2D = array(list(L2D))
		L2D.swap2D(7,9,5)
		L2D = np.asarray(L2D)
		L2D = L2D.transpose()
	if sum(sum(L2D[1:4,9:14])) < sum(sum(L2D[1:4,3:8])):
		print("no hold on") 
	
	return(L2D)	

def Coarse2(L2D):
	
	up = sum(sum(L2D[1:3,3:14]))
	down = sum(L2D[3,3:14])
	FLEt.write(str(up)+","+str(down)+",")

def Coarse4(L2D):
	
	up1 = sum(sum(L2D[1:3,3:7]))
	up2 = sum(sum(L2D[1:3,7:10]))
	up3 = sum(sum(L2D[1:3,10:14]))
	down = sum(L2D[3,3:14])
	FLEt.write(str(up1)+","+str(up2)+","+str(up3)+","+str(down)+",")
	

count = 0
for entry in signal:
    count+=1
    L0cells = getattr(entry,"L0CellEt[17][5]")
    L1cells = getattr(entry,"L1CellEt[17][5]")
    L2cells = getattr(entry,"L2CellEt[17][5]")
    L3cells = getattr(entry,"L3CellEt[17][5]")
    Hadcells = getattr(entry,"HadCellEt[17][5]") 
    seed = getattr(entry,"seed")
    eta = seed.Eta()
    phi = seed.Phi()
    viTau = getattr(entry,"mc_visibleTau")
    vipt = viTau.Pt()
    EM_Et = getattr(entry,"TauCluster_EM")
    Had_Et = getattr(entry,"TauCluster_Had")
    obsv_pt = EM_pt + Had_pt
    bcid = getattr(entry,"bcid") 
    trk = getattr(entry,"mc_tracks")
    pi0 = getattr(entry,"mc_pi0")
    L0cells_2D = Preprocess(L0cells)
    L1cells_2D = Preprocess(L1cells)
    L2cells_2D = Preprocess(L2cells)
    L3cells_2D = Preprocess(L3cells)
    Hadcells_2D = Preprocess(Hadcells)
    
    if seed.Et()>5000 and vipt>15000 and -1.4<seed.Eta()<1.4:
        Coarse2(L0cells_2D)
	    Coarse2(L1cells_2D)
	    Coarse2(L2cells_2D)
	    Coarse2(L3cells_2D)
	    FLEt.write("\n")

	



