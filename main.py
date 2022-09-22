import numpy as np
import random


def get_result(rocket_pos, rocket_speed):
    # Brute force solution
    rp = rocket_pos.copy()
    rs = rocket_speed.copy()
    n = len(rp)

    # Check if there is any chance of uniting
    def is_end(pos, speeds):
        data = [(r, s) for r, s in zip(pos, speeds)]
        sorted_arr = sorted(data, key=lambda tup: tup[0])
        _last_speed = -1
        for i in sorted_arr:
            if i[1] < _last_speed:
                return False
            _last_speed = i[1]
        return True

    # Update step
    while True:

        if is_end(rp, rs):
            break

        # update pos
        for r in range(n):
            rp[r] += rs[r]

        # check collision
        collide = []
        _ridxs = set()

        for r in range(n):
            for i in range(n - 1):
                l = i + 1
                if r + l < n:
                    if rp[r] == rp[r + l]:
                        print(f"JOIN {r} and {l} at {rp[r]}")
                        collide.append((rp[r], rs[r] + rs[r + l]))
                        _ridxs.add(r)
                        _ridxs.add(r + l)

        # remove joint rockets
        rp = [rp[c] for c in range(len(rp)) if c not in _ridxs]
        rs = [rs[c] for c in range(len(rs)) if c not in _ridxs]

        # add new rockets and speeds
        for r, s in collide:
            rp.append(r)
            rs.append(s)

        n = len(rp)

    return len(rp)


def lol():
    e = {'gemengde gebieden': 'MIXED_AREAS',
         'typisch woongebieden': 'RESIDENTIAL_AREA',
         'woongebieden met residentieel karakter': 'RESIDENTIAL_AREA',
         'landbouwgebieden': 'AGRICULTURE_AREA',
         'sterk gemengde gebieden': 'MIXED_AREAS',
         'Habitat': 'RESIDENTIAL_AREA',
         'Agricole': 'AGRICULTURE_AREA',
         'Habitat à caractère rural': 'RESIDENTIAL_AREA_WITH_RURAL_CHARACTER',
         'woongebieden': 'RESIDENTIAL_AREA',
         'agrarische gebieden': 'AGRICULTURE_AREA',
         'woongebieden met landelijk karakter': 'RESIDENTIAL_AREA_WITH_RURAL_CHARACTER',
         'woonpark': 'RESIDENTIAL_PARK',
         'landschappelijk waardevolle agrarische gebieden': 'LANDSCAPE_VALUABLE_AREA',
         'woonuitbreidingsgebieden': 'RESIDENTIAL_EXPANSION_AREA',
         'woongebieden met cultureel, historische en/of esthetische waarde': 'RESIDENTIAL_AREA_WITH_CULTURAL_HISTORICAL_AND_OR_ESTHETIC_VALUE',
         "ambachtelijke bedrijven en kmo's": 'BUSINESSES_AREA',
         'woongebieden met landelijk karakter en cultureel, historische en/of esthetische waarde': 'RESIDENTIAL_AREA_WITH_CULTURAL_HISTORICAL_AND_OR_ESTHETIC_VALUE',
         'Espaces verts': 'GREEN_SPACES',
         'parkgebieden': 'RESIDENTIAL_PARK',
         'gemengde woon- en industriegebieden': 'MIXED_RESIDENTIAL_AND_INDUSTRY_AREA',
         'bufferzones': 'BUFFER_ZONES',
         'gebieden van gewestelijk belang': 'AREAS_OF_REGIONAL_IMPORTANCE',
         'gebied voor stedelijke ontwikkeling': 'AREA_FOR_URBAN_DEVELOPMENT',
         'natuurgebieden': 'NATURAL_AREAS',
         'gebieden voor dagrecreatie': 'DAY_RECREATION_AREAS',
         'Non affecté  ("zone blanche")': 'WHITE_AREA',
         'gebieden voor gemeenschapsvoorzieningen en openbaar nut': 'COMMUNITY_AND_PUBLIC_UTILITY_AREAS',
         'milieubelastende industrieÃ«n': 'ENVIRONMENTALLY_HARMFUL_INDUSTRIES',
         'administratiegebieden': 'ADMINISTRATIVE_AREAS',
         'industriegebieden': 'INDUSTRIAL_AREAS',
         'Services publics et équipements communautaires': 'COMMUNITY_AND_PUBLIC_UTILITY_AREAS',
         'gebieden van collectief belang of van openbare diensten': 'COMMUNITY_AND_PUBLIC_UTILITY_AREAS',
         'groengebieden': 'GREEN_SPACES',
         'Forestière': 'FOREST_AREAS',
         'gebieden voor verblijfrecreatie': 'RESIDENCE_RECREATION_AREAS',
         'Parc': 'RESIDENTIAL_PARK',
         'milieubelastende industrie0n': 'ENVIRONMENTALLY_HARMFUL_INDUSTRIES',
         'recreatiegebieden': 'RECREATIONAL_AREAS',
         'natuurgebieden met wetenschappelijke waarde of natuurreservaten': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'Activité économique industrielle': 'INDUSTRIAL_AREAS',
         'bosgebieden': 'FOREST_AREAS',
         'Loisirs': 'RECREATIONAL_AREAS',
         'parkgebied voor verzorgings- instellingen': 'PARKING_AREA',
         'dienstverleningsgebieden': 'BUSINESSES_AREA',
         'uitbreiding van ontginningsgebied met nabestemming natuurontwikkeling en landbouw': 'AGRICULTURE_AREA',
         'zone voor openbaar nut met nabestemming bosgebied': 'FOREST_AREAS',
         'industriegebied met nabestemming natuurgebied (bestaande bedrijven ; geen uitbreiding mogelijk)': 'INDUSTRIAL_AREAS',
         'landbouwgebied met culturele, historische en/of esthetische waarde': 'AGRICULTURE_AREA',
         'Naturelle': 'NATURAL_AREAS',
         'groengebieden met hoogbiologische waarde': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'woonaansnijdingsgebied': 'RESIDENTIAL_AREA',
         'woonreservegebieden': 'RESIDENTIAL_AREA',
         'gebied voor kleine niet-hinderlijke bedrijven en kantoren': 'BUSINESSES_AREA',
         'gebieden hoofdzakelijk bestemd voor de vestiging van grootwinkelbedrijven': 'BUSINESSES_AREA',
         'researchpark': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'projectgebied ter uitvoering van de maatregelen tegen grondlawaai bij de economische poort internationale luchthaven Zaventem': 'BUFFER_ZONES',
         'kantoor en dienstenzone': 'BUSINESSES_AREA',
         "Centre d'enfouissement technique": 'LANDFILL_CENTER',
         'serregebieden': 'GREEN_SPACES',
         'publieke bedrijvenzones': 'BUSINESSES_AREA',
         'begraafplaatsgebieden': 'CEMETERY',
         'bouwvrij agrarisch gebied': 'AGRICULTURE_AREA',
         "Plan d'eau à créer": 'OTHER',
         None: 'OTHER',
         'gebied voor recreatiepark': 'RECREATIONAL_AREAS',
         'gemengd gemeenschapsvoorzienings- en dienstverleningsgebied (+ inrichtingen ivm haven en scheepvaart)': 'MIXED_AREAS',
         'kleintuingebied': 'GREEN_SPACES',
         'Activité économique mixte': 'MIXED_AREAS',
         'archeologische site': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'industriegebied met bijzondere bestemming (testen van autovoertuigen)': 'INDUSTRIAL_AREAS',
         'gebied voor gemeenschaps- en openbare nutsvoorzieningen in combinatie met natuurontwikkeling': 'COMMUNITY_AND_PUBLIC_UTILITY_AREAS',
         'ontginningsgebied met nabestemming natuurontwikkeling': 'FOREST_AREAS',
         'museumcentrum (in natuurgebied)': 'FOREST_AREAS',
         'kleinhandelszone': 'BUSINESSES_AREA',
         'koppelingsgebied K1/type 1': 'OTHER',
         'landelijke gebieden': 'RESIDENTIAL_AREA_WITH_RURAL_CHARACTER',
         'woongebied met recreatief karakter': 'RESIDENCE_RECREATION_AREAS',
         'gebieden voor toeristische recreatieparken (TRP)': 'LANDSCAPE_VALUABLE_AREA',
         'stortgebieden voor gepollueerde gronden (met zware metalen vervuilde grond)': 'LANDFILL_CENTER',
         'bedrijfsgebied met stedelijk karakter (kantoren, toonzalen, en in ondergeschikte orde woongelegenheid)': 'BUSINESSES_AREA',
         'water': 'OTHER',
         'stedelijk-industriegebieden': 'INDUSTRIAL_AREAS',
         'reservegebied voor bufferzone': 'BUFFER_ZONES',
         "Extraction avec destination future de zone d'espaces verts": 'GREEN_SPACES',
         'reservegebied voor sliblagunering': 'NATURAL_AREAS',
         'bufferzone met geluidswerende gebouwen bij de economische poort internationale luchthaven Zaventem': 'BUFFER_ZONES ',
         'bijzonder groengebied (cfr paardefokkerij)': 'GREEN_SPACES',
         'koninklijk domein': 'ROYAL_SPACE',
         'gebied voor zeehaven- en watergebonden bedrijven': 'INDUSTRIAL_AREAS',
         'Activité éco. spécifique agro-économique': 'AGRICULTURE_AREA',
         'uitbreidingsgbied voor stedelijke functies': 'AREA_FOR_URBAN_DEVELOPMENT',
         'gebieden voor sport- of vrijtijdsactiviteiten in de open lucht': 'RECREATIONAL_AREAS',
         'bestaande luchtvaartterreinen': 'OTHER',
         'gebieden voor jeugdcamping': 'RECREATIONAL_AREAS',
         'gebied voor duurzame stedelijke ontwikkeling': 'AREA_FOR_URBAN_DEVELOPMENT',
         'gebied voor watergebonden bedrijven': 'OTHER',
         'zone met cultuurhistorische waarde': 'RESIDENTIAL_AREA_WITH_CULTURAL_HISTORICAL_AND_OR_ESTHETIC_VALUE',
         'zone voor opslagplaatsen (bouwmaterialen zuidelijk gelegen ambachtelijke bedrijf)': 'OTHER',
         'recreatiegebied met nabestemming natuur': 'NATURAL_AREAS',
         'regionaal bedrijventerrein met openbaar karakter': 'BUSINESSES_AREA',
         "Dépendance d'extraction": 'OTHER',
         'groengebied met vissershutten': 'GREEN_SPACES',
         'zone voor natuurontwikkeling': 'NATURAL_AREAS',
         'gebied voor gemeenschapsvoorzieningen, openbare nutsvoorzieningen en natuurontwikkeling': 'COMMUNITY_AND_PUBLIC_UTILITY_AREAS',
         'zone voor jachthavenontwikkeling': 'OTHER',
         'gebied met hoofdkwartierfunctie': 'OTHER',
         'regionale gemengde zone voor diensten en handel': 'MIXED_AREAS',
         'abdijgebied': 'OTHER',
         'vliegveld / recreatie-gebied (gp Turnhout)': 'RECREATIONAL_AREAS',
         'universiteitspark': 'RECREATIONAL_AREAS',
         'recreatiepark': 'RECREATIONAL_AREAS',
         'gebieden voor de vestiging van kerninstallaties': 'NUCLEAR_INSTALLATIONS',
         'gebied voor service-residentie': 'RECREATIONAL_AREAS',
         'bufferzone met geluidswerende aarden wallen bij de economische poort internationale luchthaven Zaventem': 'BUFFER_ZONES',
         'groengebieden met semi-residentiele functie': 'RESIDENTIAL_AREA',
         'agrarische gebieden met ecologisch belang': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'natuurgebied met tijdelijke nevenfunctie waterwinning': 'NATURAL_AREAS',
         'restgebiedjes': 'RESIDENTIAL_AREA',
         'vliegveld van Deurne': 'BUFFER_ZONES',
         'bosgebieden met ecologisch belang': 'FOREST_AREAS',
         'bijzondere industriegebieden (afvalverwerking)': 'LANDFILL_CENTER',
         'gebieden voor dag- Ã©n verblijf- recreatie': 'RECREATIONAL_AREAS',
         'militaire gebieden': 'MILITARY_AREA',
         'transportzone': 'OTHER',
         'gebied voor uitbreiding van bestaande nijverheid': 'INDUSTRIAL_AREAS',
         'gebied voor handelbeursactiviteiten en grootschalige culturele activiteiten': 'BUSINESSES_AREA',
         'ondernemings gebieden in de stedelijke omgeving': 'BUSINESSES_AREA',
         'zone voor handelsvestigingen': 'BUSINESSES_AREA',
         'bestaande autosnelwegen': 'OTHER',
         'gebied voor natuureducatieve infrastructuur': 'NATURAL_AREAS',
         'lokaal bedrijventerrein met openbaar karakter': 'BUSINESSES_AREA',
         'ontginningsgebied met nabestemming recreatie en natuur': 'RECREATIONAL_AREAS',
         'koppelingsgebied K 2/type 2': 'OTHER',
         'wetenschapspark': 'RECREATIONAL_AREAS',
         'gebieden voor wachtbekken': 'OTHER',
         'Activité eco. spécifique grande distribution': 'BUSINESSES_AREA',
         'gebied voor gemeenschapsvoorzieningen en openbare nutsvoorzieningen met nabestemming natuurgebied met wetenschappelijke waarde of natuurreservaat (enkel instandhouding of gezondmaking bestaand gebouwencomplex)': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'recreatieve parkgebieden': 'RECREATIONAL_AREAS',
         'industrie-stortgebied (niet-giftige industriÃ«le afval)': 'LANDFILL_CENTER',
         'natuurgebied met bijzondere voorschriften voor de kleinijverheid': 'NATURAL_AREAS',
         'luchthaven gebonden bedrijventerrein': 'BUSINESSES_AREA',
         'zone voor Koninklijk Domein': 'ROYAL_SPACE',
         'gebieden voor havenactiviteiten en vervoeren': 'OTHER',
         'gebied voor wachtbekken met ondergeschikte waterrecreatieve functie': 'OTHER',
         'pleisterplaats voor nomaden of woonwagenbewoners': 'OTHER',
         'industriegebied met nabestemming woongebied': 'INDUSTRIAL_AREAS',
         'parkgebieden met semi-agrarische functie': 'GREEN_SPACES',
         'Aménagement communal concerté': 'OTHER',
         'milieubelastende industrieën': 'ENVIRONMENTALLY_HARMFUL_INDUSTRIES',
         "valleigebieden (of 'agrarische gebieden met landschappelijke waarde')": 'AGRICULTURE_AREA',
         'natuureducatieve infrastructuur': 'OTHER',
         'bijzondere natuurgebieden (waterzuivering, afvoerleidingen en leidingstraten)': 'NATURAL_AREAS',
         'gebied voor kernontwikkeling': 'AREA_FOR_URBAN_DEVELOPMENT',
         'zone voor kleinhandel en kleine en middelgrote ondernemingen': 'BUSINESSES_AREA',
         'gebied voor uitbreiding en sanering van bestaande nijverheid': 'INDUSTRIAL_AREAS',
         'spoorweggebieden': 'OTHER',
         'natuurgebied met erfdienstbaarheid (t.a.v. transport- en pijpleidingen)': 'NATURAL_AREAS',
         'tijdelijk gebied voor gemeenschapsvoorzieningen (autokeuring)': 'OTHER',
         'grondreservegebieden': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'oeverstrook met bijzondere bestemming (Antwerpse kaaien)': 'OTHER',
         "Plan d'eau": 'OTHER',
         'zone voor gemeenschaps- en openbare nutsvoorzieningen met nabestemming natuurgebied': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'industriegebied voor milieubelastende industrie met nabestemming groengebied (breekwerf- en betoncentrale)': 'ENVIRONMENTALLY_HARMFUL_INDUSTRIES',
         'bestaande waterwegen': 'OTHER',
         'teleport (hoogwaardig kantorenpark met geavanceerde telecommunicatievoorzieningen)': 'OTHER',
         'bijzonder reservatiegebied (cfr Teleport)': 'NATURAL_AREAS_OF_SCIENTIFIC_VALUE_OR_NATURE_RESERVES',
         'gebied voor luchthavengerelateerde kantoren en diensten': 'OTHER',
         'Vierge de toute affectation (annulation du Conseil Etat)': 'OTHER',
         'golfterrein': 'RECREATIONAL_AREAS',
         'stortgebied met nabestemming natuurontwikkeling (niet-giftige baggerspecie)': 'LANDFILL_CENTER',
         "agrarisch gebied met landschappelijke (of 'bijzondere') waarde (vallei- of brongebieden)": 'AGRICULTURE_AREA',
         'Aménagement communal concerté à caractère économique': 'BUSINESSES_AREA',
         'business-park (vergader- en congresactiviteiten + overnachting, restaurant)': 'BUSINESSES_AREA',
         'Extraction avec destination future de zone naturelle': 'NATURAL_AREAS'}

    return set([c for c in e.values()])


if __name__ == '__main__':
     for i in lol():
          print(f"{i} = '{i}'")
