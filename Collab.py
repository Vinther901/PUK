import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from ExternalFunctions import Chi2Regression


class Dataset():
    def __init__(self, path, params = None, masstype = "ks"):
        """
        Initialize the DataSet, the initial conditions will be written in later
        """
        self.path = path
        self.params = params
        self.masstype = masstype
    
    def load(self, load_amount, test_ratio, mass_range = None):
        """
        Load data from a root file
        """

        # Import uproot and load file
        import uproot
        file = uproot.open(path)
        data = file['tree'].pandas.df(params, entrystop = load_amount)
        mass = file['tree'].pandas.df("v0_mass_" + self.masstype, entrystop = load_amount)
        try:
            label = file['tree'].pandas.df("true" + self.masstype.capitalize(), entrystop = load_amount)
        except:
            pass


        # Apply a mass range limit. 
        if mass_range:
            mask = (mass >= mass_range[0]) & (mass <= mass_range[1])
            data = data.loc[mask, :]
            mass = mass.loc[mask, :]
            if label:
                label = label[mask]
            self.bins = int((mass_range[1] - mass_range[0])/0.5)


        # Split set
        from sklearn.model_selection import train_test_split
        if label:
            df_train, df_test, train_label, test_label, train_mass, test_mass = \
                train_test_split(data, mass, label, test_size = test_ratio)
        else:
            df_train, df_test, train_mass, test_mass = \
                train_test_split(data, mass, test_size = test_ratio)
            train_label, test_label = None, None
        
        # Define dataset
        self.train_set = df_train
        self.train_mass = train_mass
        self.test_set  = df_test
        self.test_mass = test_mass
        self.train_label = train_label
        self.test_label = test_label
    
    def fit_test_mass(self, mask = None, plot = False):
        if not mask:
            vals, binedges = np.histogram(self.dataset.test_mass, self.dataset.bins)
        else:
            vals, binedges = np.histogram(self.dataset.test_mass[mask], self.bins)
        binc = 0.5 * (binedges[:-1] + binedges[1:])
        
        #FIT

        #self.sig = (sig_count,sig_err)
        #self.bkgr = (bkgr_count, bkgr_err)
    
    def fit_mass(self, mass, ax = None, double=1, poly_degree=3, depth=50, plot=True):
        """Gauss fit. If double we fit with mu_1=mu_2, with polynomial backgground fit:
        Returns fig, ax, the full Minuit object, background amount and signal amount:
        """
        vals, binedges = np.histogram(mass,bins=self.bins,range=self.mass_range)
        xs = 0.5*(binedges[:-1] + binedges[1:])
        mask = vals > 0
        xs, vals, errs = xs[mask], vals[mask], np.sqrt(va)[mask]
        #look into automatizing the guesses
        mu, sigma, N = guesses_sig
        #Make a 5 sigma cut so only background is here
        bkgr_mask = (xs < mu-5*sigma) | (xs > mu+5*sigma)
        guesses_bkgr=np.zeros(poly_degree+1)
        guesses_bkgr[-1], guesses[-2] = (vals[0]+vals[-1])/2, (vals[-1]-vals[0])/bins
        def background_fit(x, a, b, c, d):
            return a * (x-mu) ** 3 + b * (x-mu) ** 2 + c * (x-mu) + d
         # Background fit under here
        if sum(bkgr_mask])<poly_degree+1
            def background_fit(xs, a, b, c, d):
                return 0*xs
            b1, b2, b3, b4=0,0,0,0
        else:
            vals_b, cov_b = curve_fit(background_fit, xs[bkgr_mask], vals[bkgr_mask], p0 = guesses_bkgr)
            b1, b2, b3, b4 = vals_b
            bkgr_chi2 = Chi2Regression(background_fit, xs[bkgr_mask], vals[bkgr_mask], errs[bkgr_mask])
            bkgr_min  = Minuit(bkgr_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4, limit_d=[0,2*b4])
            bkgr_min.migrad()
            counter = 0
            bkgr_min
            while not bkgr_min.valid and counter<depth:
                bkgr_min.migrad()
                counter += 1
            if not bkgr_min.valid: print("No background valid minimum found!")

        #Save guesses 
        b1, b2, b3, b4 = bkgr_min.args

        # The signal fit  Here gauss
        def gauss(x, mean, sig, size):
            return size*norm.pdf(x, mean, sig)

        # def gauss2(x, mean, sig, size):
        #     return size*norm.pdf(x, mean, sig)
        
        if double:
            # Full fit for double gauss
            def full_fit(x, mean, sig, size, f, sigmp, a, b, c, d):
                return background_fit(x, a, b, c, d) + f*gauss(x, mean, sig, size) + (1-f)*gauss(x, mean, sigmp*sig, size)
        else:
            # Full fit for double gauss
            def full_fit(x, mean, sig, size, a, b, c, d):
                return background_fit(x, a, b, c, d) + gauss(x, mean, sig, size)

class Model():
    
    def __init__(self, classifier, dataset):
        self.classifer = classifier
        self.dataset = dataset

    def train():
        self.classifier.fit(dataset.train_set,dataset.train_set.y)
    
    




    

    


