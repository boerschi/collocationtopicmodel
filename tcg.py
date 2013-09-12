import random, math, sys

"""
  Breakpoint sampler for Mark Johnson's Topical Collocation Model by benjamin boerschinger
  
  date: 11/09/13
"""
  
MAX_DISPLAY=20



def writeSample(m,outf=sys.stdout):
    """write the current analysis of sampler m to outf"""
    for i in range(len(m.sents)):
        s = m.sents[i]
        dId = m.docInds[i]
        b = m.sentsbounds[i]
        res = []
        for j in range(len(s)-1):
            if b[j]!=0:
                (c1,c2,c12)=m.getCollocsMid(j,s,b)
                res.append("(Topic%s %s)"%(b[j]," ".join(c1)))
        res.append("(Topic%s %s)"%(b[-1]," ".join(m.getFinal(s,b))))
        outf.write("%s %s\n"%(dId," ".join(res)))

class WordGen(object):
    """base-distribution over the fixed vocabulary, fixed for simplicity but could be parametrized by draw from Dirichlet
    
    if you decide to put a prior on the parameter on WordGen, rather than having it be uniform, the model becomes hierarchical
    and you need to track tables in the individual Topic-Restaurants, as the number of tables with label X across all restaurants
    correspnds to the number of Counts in the base-distribution Predictive-Posterior
    """

    def __init__(self,V,p_cb):
        """V is number of words, p_cb probability of stopping
        
        setting p_cb small encourages long collocations, setting it high penalizes long collocations
        """
        self.pw = 1/float(V)
        self.p_cb = p_cb
    
    def __call__(self,c):
        """c is a tuple of words"""
        return (self.pw*(1-self.p_cb))**len(c) * (self.p_cb/(1-self.p_cb))

class CRP(object):
    """a simple CRP class, used for the different Topics"""
    
    def logProb(self):
        res = math.lgamma(self.alpha)-math.lgamma(self.alpha+self.n)
        for (w,nk) in self.counts.iteritems():
            res = res+math.lgamma(nk+self.alpha)-math.lgamma(self.alpha)
        return res

    def __init__(self,alpha,base):
        """setting alpha small will force few collocations to be used within each topic, setting it high will allow more collocation types"""

        self.base = base
        self.alpha=alpha
        self.n=0
        self.counts = {}

    def count(self,w):
        try:
            return self.counts[w]
        except KeyError:
            return 0.0
    
    def add(self,w):
        self.counts[w]=self.count(w)+1
        self.n+=1
    
    def remove(self,w):
        newCount = self.count(w)-1
        self.n-=1
        if newCount==0:
            self.counts.pop(w)
        elif newCount>0:
            self.counts[w]=newCount
        else:
            raise Exception("Can't remove %s, not present."%w)

    def __call__(self,w):
        return (self.count(w)+self.alpha*self.base(w))/(self.alpha+self.n)

    def __repr__(self):
        global MAX_DISPLAY
        res = "# CRP object with %s customers of %s types\n"%(self.n,len(self.counts.keys()))
        displayed = 0
        for (w,c) in sorted(self.counts.iteritems(),lambda x,y:-cmp(x[1],y[1])):
            res += "# P(%s)=%.3f\n"%(w,self(w))
            displayed+=1
            if displayed>MAX_DISPLAY:
                break
        return res

    def __str__(self):
        return self.__repr__()


class PostPredDir(object):
    """Posterior Predictive for an integrated out Draw from Dirichlet"""

    def __init__(self,alpha,K):
        self.counts = {}
        self.norm = alpha*len(K)
        self.alpha = alpha
        self.n=0
        for o in K:
            self.counts[o]=alpha
    
    def __repr__(self):
        res = "# DirPostPred object with %s outcomes of %s types\n"%(self.n,len(self.counts.keys()))
        for w in self.counts.keys():
            res += "# P(%s)=%.3f\n"%(w,self(w))
        return res

    def __str__(self):
        return self.__repr__()

    def remove(self,o):
        self.counts[o]-=1
        self.n-=1
    
    def add(self,o):
        self.counts[o]+=1
        self.n+=1

    def __call__(self,o):
        return self.counts[o]/(self.n+self.norm)
    
    def logProb(self):
        res = math.lgamma(self.norm)-math.lgamma(self.norm+self.n)
        for k in self.counts.values():
            res = res + math.lgamma(k)-math.lgamma(self.alpha)
        return res


class TCG(object):
    """a Topical Collocation Model Gibbs-Sampler"""

    def getCollocsMid(self,pos,sent,bounds):
        """return the info for a mid-boundary, a 3-tuple of (c1,c2,c1c2)"""
        start=pos-1
        while bounds[start]==0 and start!=-1:
            start-=1
        end=pos+1
        while bounds[end]==0 and end!=len(sent):
            end+=1
        c1 = sent[start+1:pos+1]
        c2 = sent[pos+1:end+1]
        c1c2 = sent[start+1:end+1]
        return (c1,c2,c1c2)

    def getFinal(self,sent,bounds):
        """return final collocation"""
        start=len(sent)-2
        while bounds[start]==0 and start!=-1:
            start-=1
        return sent[start+1:]

    def logProb(self):
        res = sum([x.logProb() for x in self.topics])+sum([x.logProb() for x in self.docMixes.values()])
        return res
    
    def initCounts(self):
        """initialize the distributions with the random initialization"""
        for (docId,s,b) in zip(self.docInds,self.sents,self.sentsbounds):
            for i in range(len(s)-1):
                if b[i]!=0:
                    self.docMixes[docId].add(b[i]) #increment topic
                    (c1,c2,c1c2) = self.getCollocsMid(i,s,b)
                    self.topics[b[i]].add(c1)
            self.topics[b[-1]].add(self.getFinal(s,b))
            self.docMixes[docId].add(b[-1])

    def writeSample(self,outf=sys.stdout):
        """write the current analysis to outf"""
        writeSample(self,outf)
              
    def __init__(self,data,ntopics=4,alpha_theta=0.1,alpha_phi=0.1,p_stop=0.5,p_cb=0.5):
        """initialize sampler using data

        data has to have one sentence per line, preceded by a document identifier
        ntopics is the number of topics used
        alpha_theta is the uniform pseudo-count for the document-mixtures
        alpha_phi is the concentration parameter for the non-parametric topics
        p_stop is the stopping probability for generating a sentence in a document
        p_cb is the stopping probability for a collocation in the base-distribution

        intuitively, a high p_stop should lead to fewer items per sentence, i.e. longer collocations
        the inverse holds for p_cb, it encourages long collocations in the base-distribution for small p_cb, and discourages them for high p_cb

        you can put priors on all these parameters but I wouldn't necessarily expect that to increase quality in any measurable sense, although it
        gets around the hard issue of setting them by hand (e.g. running a grid-search on smaller data...)
        """

        self.K = ntopics
        self.sents = []    #sentences
        self.sentsbounds = []
        self.docInds = []  #document to which each sentence belongs
        self.p_stop = p_stop
        self.p_cb = p_cb
        wordCounts = {}
        for l in open(data):
            l = l.strip()
            docId,words = l.split(" ",1)
            self.docInds.append(docId)
            self.sents.append(tuple(words.split()))
            for w in self.sents[-1]:
                wordCounts[w]=1
        self.docMixes = { x:PostPredDir(alpha_theta,[y for y in range(1,self.K+1)]) for x in set(self.docInds) } #document mixtures
        self.V = len(wordCounts.keys()) #base-vocabulary
        self.base = WordGen(self.V,p_cb) #base-distribution over base-vocabulary
        self.topics = [CRP(alpha_phi,self.base) for x in range(self.K+1)] #the non-parametric topic-mixtures, index0 is empty, to make indexation easier
        
                           
        # initialize the sent-analyses, last indicator has to  be a topic, must not be 0
        for s in self.sents:
            n = len(s)
            res = []
            for i in range(n-1):
                res.append(random.randint(0,self.K))
            res.append(random.randint(1,self.K))
            self.sentsbounds.append(res)

        self.initCounts()

    def resampleSent(self,j):
        """resample the jth sentence"""
        docId = self.docInds[j]
        s = self.sents[j]
        b = self.sentsbounds[j]
        for i in range(len(s)-1):
            (c1,c2,c1c2)=self.getCollocsMid(i,s,b)
            ti1 = b[i]
            ti2 = b[i+len(c2)]
            #remove counts, if boundary was present, c1 and c2, else c1c2
            if b[i]==0:
                self.topics[ti2].remove(c1c2)
                self.docMixes[docId].remove(ti2)
            else:
                self.topics[ti1].remove(c1)
                self.topics[ti2].remove(c2)
                self.docMixes[docId].remove(ti1)
                self.docMixes[docId].remove(ti2)
            isFinal = i+len(c2)==len(b)
            
            #outcomes is an unnormalized distribution over all possible values for the boundary, 0 up to K
            outcomes = [self.pNoBound(c1c2,ti2,docId,isFinal)]
            for k in range(1,self.K+1):
                outcomes.append(self.pBoundT(c1,c2,k,ti2,docId,isFinal))
            
            # sample from this distribution and update sampler state
            new = self.sample(outcomes)
            if new==0:
                self.topics[ti2].add(c1c2)
                self.docMixes[docId].add(ti2)
            else:
                self.topics[new].add(c1)
                self.docMixes[docId].add(new)
                self.topics[ti2].add(c2)
                self.docMixes[docId].add(ti2)
            b[i]=new
        # the last boundary can only range over 1 to K, but the rest is analogous
        cLast = self.getFinal(s,b)
        self.topics[b[-1]].remove(cLast)
        self.docMixes[docId].remove(b[-1])
        outcomes = [0]
        for k in range(1,self.K+1):
            outcomes.append(self.pLastT(cLast,k,docId))
        new = self.sample(outcomes)
        b[-1]=new
        self.topics[new].add(cLast)
        self.docMixes[docId].add(new)
    
    def sweep(self):
        """perform a single Gibbs sweep over all sentences in random order"""
        inds = range(len(self.sents))
        random.shuffle(inds)
        for i in inds:
            self.resampleSent(i)
        print self.logProb()

    def pLastT(self,c,t,dId):
        """probability of setting the final topic-label to t"""
        return self.docMixes[dId](t)*self.topics[t](c)

    def pNoBound(self,c1c2,t,dId,isFinal):
        """probability of positing the long collocation c1c2 under topic t in document dId"""
        res = self.docMixes[dId](t)*self.topics[t](c1c2)
        if isFinal:
            return res*self.p_stop
        else:
            return res*(1-self.p_stop)

    def pBoundT(self,c1,c2,t1,t2,dId,isFinal,exact=False):
        """probability of positing collocation c1 under t1 and c2 under t2 in dId

        strictly speaking, we need to perform an intermediate update after calculating the probabilities for c1
        this can induce noticeable overhead and shouldn't make a big difference for large data
        for doing the proper update, set exact to True
        """
        res = self.docMixes[dId](t1)*self.topics[t1](c1)*(1-self.p_stop)
        if exact:
            self.docMixes[dId].add(t1)
            self.topics[t1].add(c1)
        res = res*self.docMixes[dId](t2)*self.topics[t2](c2)
        if isFinal:
            res = res*self.p_stop
        else:
            res = res*(1-self.p_stop)
        if exact:
            self.docMixes[dId].remove(t1)
            self.topics[t1].remove(c1)
        return res

    def sample(self,l):
        """perform a sample from an unnormalized distribution, return the index of the outcome that was sampled"""
        norm = sum(l)
        flip = random.random()*norm
        cur = 0
        for (i,p) in enumerate(l):
            cur+=p
            if flip<=cur:
                return i
        raise Exception("Couldn't sample from %s"%l)

if __name__=="__main__":
    print "using 10 Topics, running for 10 iterations, writing to %s"%sys.argv[2]
    model = TCG(sys.argv[1],10)
    outf = open(sys.argv[2],"w")
    for i in range(10):
        model.sweep()
        model.writeSample(outf)
