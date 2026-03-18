class Classe_VCT:
    dict_valeur_force = {}
    """ Classe de base qui permet de gerer l'affichage en python de nos classes"""
    def __init__(self, args={}, **kwargs):
        self.__dict__.update(args)
        self.__dict__.update(kwargs)
        self._setup()

    def _setup(self):
        """ Fonction setup, qui après tous les calculs, impose les valeurs forcées (utile pour les interfaces)"""
        #Set à partir du dict de valeurs
        for key, value in self.dict_valeur_force.items():
            setattr(self, key, value)


    def set(self, args={}, **kwargs):
        """ Fonction set de l'objet """
        # Gestion du set sur un attribut d'attribut-objet exemple : si on set [self.truc.machin = 10]
        liste_a_virer = []  #Liste les keys à virer
        for key, value in args.items():    #Récup l'objet-attribut et set son attribut
            if "." in key:
                # sépare l'attribut-objet de son attibut
                liste_attribut = key.split(".")
                object_str = liste_attribut[0]
                attribut_str = ".".join(liste_attribut[1:])
                # set l'attribut-objet (Recusivité possible)
                obj =  self.get(object_str)
                obj.set({attribut_str : value})

                liste_a_virer.append(key)
        #
        # pop les keys qui ont été settées
        for key in liste_a_virer : args.pop(key)
        #
        # Set le reste
        self.__dict__.update(args)
        self.__dict__.update(kwargs)
        self._setup()

    def _setParentUp(self, args={}, **kwargs):
        """ Fonction qui permet de setter les parents en plus de l'objet """
        self.set(args, **kwargs)
        # setup le parent si il existe
        if 'parent' in self.__dict__ and '_setup' in self.parent.__dir__():
            self.parent._setParentUp()

    def deepSet(self, eviterRecursion=None, args={}, **kwargs):
        """Fonction qui vise à setter l'ensemble des attributs de l'objet en évitant les récusion. CETTE FONCTION N'EST PAS ENCORE VALIDEE"""
        self.set(args, **kwargs)
        if eviterRecursion == None: eviterRecursion=[]   # evite de modifier la liste en entrée si on introduit [eviterRecursion=[] directement, la fonction ne marche pas]
        eviterRecursion.append(self) # Permet d'éviter la récursion lorsque deux objets sont attributs mutuels
        # Recherche l'ensemble des attributs qui sont des objets Classe_VCT
        for attribut in self.__dict__:
            try: # évite les erreurs avec les classes de np.array
                if self.__dict__[attribut] not in eviterRecursion and issubclass(self.__dict__[attribut].__class__, Classe_VCT):
                    self.__dict__[attribut].deepSet(eviterRecursion)
            except:
                pass

    def get(self, attribut_str=None, eviterRecursion=None, deep=False):
        """ Fonction qui recherche l'attribu cité, dans le dictionnaire de la classe, mais aussi dans le dictionnaire des enfants de l'objet (sauf le parent)"""
        #Si il n'y a pas d'attribut demandé, revoie l'objet
        if attribut_str is None:
            return self
        #
        # Si l'attribut demandé est l'attribut d'un attribut-bojet de self
        if "." in attribut_str: # gère le cas de setter indirect
            # sépare l'attribut-objet de son attibut
            liste_attribut = attribut_str.split(".")
            object_str = liste_attribut[0]
            attribut_str = ".".join(liste_attribut[1:])
            # get l'attribut-objet (Recusivité possible)
            obj =  self.get(object_str)
            return obj.get(attribut_str, eviterRecursion, deep)
        #
        out = None #sortie
        if eviterRecursion == None: eviterRecursion=[] # evite de modifier la liste en entrée
        eviterRecursion.append(self)    # Permet d'éviter la récursion lorsque deux objets sont attributs mutuels
        # Si l'attribu demandé est attribut de l'objet
        try:
            out = getattr(self, attribut_str)
        except AttributeError:
            if deep:
                for attribut in self.__dict__:
                    try: # évite les erreurs avec les classes de np.array
                        # Si on à pas déjà cherché dans l'attribu
                        if self.__dict__[attribut] not in eviterRecursion:
                            # Si l'attribut a pour subclass Classe_VCT on peut faire appel à son getter
                            if issubclass(self.__dict__[attribut].__class__, Classe_VCT) and attribut != 'parent':
                                out = self.__dict__[attribut].get(attribut_str, eviterRecursion)
                                if out is not None :
                                    break
                    except:
                        print("Erreur :", sys.exc_info()[1])
        except:
            print("Erreur :",sys.exc_info()[1])

        return out

    def copy(self, KeepParent=True, **kwargs):
        # circule dans le dictionnaire et récupère des copies des objets attributs
        copy = self.__dict__.copy()
        for nomAttribut in copy:
            attribut = copy[nomAttribut]
            # dans le cas d'une liste, renvoie la liste des copie
            if type(attribut) is list:
                newAttribut = []
                for element in attribut:
                    if issubclass(element.__class__, Classe_VCT):
                        newAttribut.append(element.copy(KeepParent=True))
                    elif 'copy' in element.__dir__() :
                        newAttribut.append(element.copy())
                    else:
                        newAttribut.append(element)
                copy[nomAttribut] = newAttribut
            # ne copy l'attribu qui si c'est possible, sauf si c'est le parent de l'objet (évite les recursion)
            elif nomAttribut=='parent' :
                pass
            elif issubclass(copy[nomAttribut].__class__, Classe_VCT):
                copy[nomAttribut] = attribut.copy(KeepParent=True)
            elif 'copy' in attribut.__dir__() :
                copy[nomAttribut] = attribut.copy()

        if ('parent' in copy) and KeepParent: copy.pop('parent')

        return self.__class__(args=copy, **kwargs)

    def enregistrer(self, chemin, nom_fichier="nom_fichier"):
        import pickle
        """ fonction qui utilise la lib pickle pour save l'objet"""
        pickle.dump(self, open(chemin + "\\" + nom_fichier + ".pkl", 'wb'))


    def __str__(self):
        return '<Class '+ self.__class__.__name__ + '>'

    def __repr__(self):
        dictionnaire = self.__dict__
        keys = sorted(dictionnaire.keys())
        stringOut = '<Class '+ self.__class__.__name__ + ' :\n'
        for key in keys:
            stringOut = stringOut + '   .'+ str(key) + ' : '
            #Si c'est un float l'arrondi
            if type(dictionnaire[key]) is float:
                stringOut = stringOut + str(dictionnaire[key]) + '\n'
            # si c'est une liste affiche la liste des affichages simplifiiés
            elif type(dictionnaire[key]) is list:
                stringOut = stringOut + '['
                for element in dictionnaire[key]:
                    stringOut = stringOut + ' ' + element.__str__() + ','
                stringOut = stringOut + ']' + '\n'
            # Si c'est un dictionnaire affiche simplement un représentation <class 'dict'>
            elif type(dictionnaire[key]) is dict:
                stringOut = stringOut + "<class 'dict'" + ' >\n'
            # Sinon
            else:
                # Si c'est un attirbut qui peut etre mesuré:
                if '__len__' in dictionnaire[key].__dir__():
                    # si la longueur de l'element permet de le visualiser
                    if dictionnaire[key].__len__() < 50:
                        stringOut = stringOut + dictionnaire[key].__str__() + '\n'
                    else:
                        stringOut = stringOut + str(type(dictionnaire[key])) + '\n'
                else:
                    stringOut = stringOut + dictionnaire[key].__str__() + '\n'

        stringOut = stringOut + 'End '+ self.__class__.__name__ + ' >\n'
        return str(stringOut)

    def set_force(self, attribut_str, valeur=None):
        """ Fonction qui permet de forcer une valeur, et la setter sur l'attribut, fonctionne avec les attribut-objets """
        #  Gestion du set sur un attribut d'attribut-objet exemple : si on set [self.truc.machin = 10]
        if "." in attribut_str: # gère le cas de setter indirect
            # sépare l'attribut-objet de son attibut
            liste_attribut = attribut_str.split(".")
            object_str = liste_attribut[0]
            attribut_str = ".".join(liste_attribut[1:])
            # set_force l'attribut-objet (Recusivité possible)
            obj =  self.get(object_str)
            obj.set_force(attribut_str, valeur)
        else:
            # Foce la valeur
            if valeur :
                self.dict_valeur_force[attribut_str] = valeur
                self.set({attribut_str: valeur})
            # pop la valeur forcée
            else:
                try:
                    self.dict_valeur_force.pop(attribut_str)
                except:
                    pass


    def is_force(self, attribut_str):
        """ Check si une valeur est forcée, fonctionne avec les attribuut-objets"""
        if "." in attribut_str: # gère le cas de setter indirect
            # sépare l'attribut-objet de son attibut
            liste_attribut = attribut_str.split(".")
            object_str = liste_attribut[0]
            attribut_str = ".".join(liste_attribut[1:])
            # is_force l'attribut-objet (Recusivité possible)
            obj =  self.get(object_str)
            return obj.is_force(attribut_str)
        else :
            return (attribut_str in self.dict_valeur_force)