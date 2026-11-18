"""Generic, industry-independent procurement vocabulary.

This is NOT a list of any company's categories. It is general purchasing language: synonyms
(including common translations and ERP abbreviations) and broader words that purchasing
category trees tend to use. The classifier reads the company's tree at run time and uses this
vocabulary only to bridge the gap between how invoice lines are written ("BRG 6205 2RS",
"rodamiento", "Misc Freight / Courier") and how category names are written.

Format of each line:   head = synonym, synonym, ... | broader, broader, ...
All words are lower-case, accent-free, singular. One line per concept.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_RAW = r"""
# ---------------- MRO / mechanical
bearing = bearings, brg, brng, rodamiento, rolamento, lager, kugellager, roulement, cuscinetto, lozisko, rulman, bush, bushing | mechanical, spare, part, component, maintenance, mro, repair
seal = seals, oring, gasket, junta, joint, dichtung, guarnizione, retentor, packing, sello | mechanical, spare, part, maintenance, mro
valve = valves, vlv, valvula, ventil, vanne, valvola, klapan | mechanical, spare, part, fitting, hydraulic, pneumatic, pump, mro
pump = pumps, pmp, bomba, pumpe, pompe, pompa, impeller | mechanical, spare, part, equipment, maintenance, mro
motor = motors, mtr, gearmotor, servomotor, moteur, motore | electrical, mechanical, spare, part, drive, automation, mro
gear = gears, gearbox, reducer, reductor, getriebe, engrenage, sprocket, pinion, coupling, cardan | mechanical, spare, part, drive, transmission, mro
shaft = shafts, axle, eje, welle, arbre, albero | mechanical, spare, part, mro
belt = belts, correa, riemen, courroie, cinghia, vbelt | mechanical, spare, part, conveyor, transmission, mro
chain = chains, cadena, kette, chaine, catena | mechanical, spare, part, transmission, mro
spring = springs, resorte, feder, ressort, molla | mechanical, spare, part, mro
roller = rollers, idler, rodillo, rolle, rouleau, rullo | mechanical, spare, part, conveyor, mro
fastener = fasteners, screw, screws, nut, nuts, washer, washers, rivet, stud, tornillo, tuerca, arandela, schraube, mutter, vis, ecrou, vite, dado, parafuso, porca, pin, clip, circlip, dowel | hardware, mechanical, spare, part, consumable, mro
bolt = bolts, perno, bolzen, boulon, bullone | fastener, hardware, mechanical, spare, part
spare = spares, sparepart, repuesto, refaccion, ersatzteil, piece, ricambio, peca, kit, assembly, assy, component, part | mechanical, maintenance, mro
part = parts, pieza, teil, piece, parte, componente, component | spare, mechanical, mro
mechanical = mech, mecanico, mechanisch, mecanique, meccanico | spare, part, maintenance, mro
hydraulic = hyd, hydr, hidraulico, hydraulik, hydraulique, idraulico, cylinder, cilindro, zylinder | hose, fitting, mechanical, spare, part, mro
pneumatic = pneu, pneum, neumatico, pneumatik, pneumatique, pneumatico, actuator | hose, fitting, mechanical, spare, part, mro
hose = hoses, manguera, schlauch, tuyau, flexible, tubo, mangueira, tubing, tube, pipe, piping | fitting, pneumatic, hydraulic, plumbing, mro
fitting = fittings, coupling, connector, conector, conexion, raccord, raccordo, adapter, adaptor, elbow, tee, nipple, flange, brida, flansch, bride, union, manifold | hose, pneumatic, hydraulic, plumbing, pipe, mro
filter = filters, fltr, flt, filtro, filtre, cartridge, cartucho, strainer, element, micron | mro, consumable, spare, part, maintenance
lubricant = lubricants, lube, lub, oil, oils, grease, aceite, grasa, oel, schmierstoff, huile, graisse, olio, grasso, oleo, lubrificante, lubricante, hydraulic oil, mobil, shell, castrol, tellus | lubricant, oil, chemical, consumable, mro, maintenance
tool = tools, tooling, herramienta, werkzeug, outil, utensile, ferramenta, wrench, spanner, saw, blade, knife, hammer, plier, screwdriver, socket, grinder, disc, abrasive, toolroom | shop, consumable, hardware, mro, workshop
consumable = consumables, consumible, verbrauchsmaterial, consommable, consumo, tape, cinta, adhesive, glue, rag, wipe, brush, sandpaper, cable tie | shop, tool, supply, mro, general
welding = weld, welder, soldadura, schweissen, soudure, saldatura, electrode, wire | consumable, tool, workshop, mro, gas
paint = paints, coating, pintura, farbe, peinture, vernice, primer, enamel, lacquer, laca | coating, chemical, consumable, maintenance
# ---------------- electrical / automation
electrical = elec, elect, electric, electrico, elektrisch, electrique, elettrico, electrical, electro | electrical, automation, spare, part, cable, mro
cable = cables, cbl, cabling, cable, kabel, cavo, cabo, wire, wiring, conductor | electrical, spare, part, installation
switch = switches, switchgear, swgr, breaker, contactor, relay, rele, fuse, fusible, sicherung, interruptor, schalter, interrupteur, isolator, disconnect | electrical, automation, spare, part, distribution
sensor = sensors, transmitter, encoder, proximity, photoelectric, photoelec, detector, probe, thermocouple, gauge, manometer, instrument, instrumentation | electrical, automation, control, spare, part
automation = plc, hmi, drive, vfd, inverter, servo, controller, siemens, allen, bradley, rockwell, sinamics, fanuc, control, controls, scada | electrical, automation, spare, part, it
lighting = light, lights, lamp, lamps, bulb, led, luminaire, fixture, lampara, leuchte, lampe | electrical, facility, maintenance
battery = batteries, bateria, batterie, accumulator, ups, charger | electrical, spare, part
transformer = transformers, transformador, transformateur, substation, substations | electrical, power, equipment
generator = generators, genset, generador | electrical, power, equipment, energy
# ---------------- metals and raw materials
aluminium = aluminum, alu, aluminio, aluminium, alluminio, alumínio, ingot, billet, alloy | metal, raw, material, direct
steel = steels, stl, acero, stahl, acier, acciaio, aco, tinplate, stainless, galvanised, galvanized | metal, raw, material, direct
metal = metals, metallic, metal, metall, metallo, scrap, coil, coils, sheet, plate, strip, slab | metal, raw, material, direct, aluminium, steel
copper = cobre, kupfer, cuivre, rame, brass, bronze | metal, raw, material
coil = coils, bobina, coil, rolle, bobine | metal, stock, raw, material, sheet
resin = resins, polymer, polymers, plastic, plastics, pellet, granulate, pet, hdpe, ldpe, pp, pvc, polyethylene, polypropylene, resina, kunststoff | raw, material, chemical, packaging
chemical = chemicals, chem, quimico, quimicos, chemikalie, chimique, chimico, solvent, reagent, acid, caustic, additive, compound, catalyst | chemical, raw, material, consumable
coating = coatings, coat, compound, compounds, sealant, lining, lacquer, varnish | coating, chemical, direct, material
ink = inks, tinta, tinte, encre, inchiostro, toner, pigment, dye, colorant | ink, coating, print, consumable
varnish = varnishes, barniz, lack, vernis, vernice, lacquer, laca | coating, ink, chemical
paper = papel, papier, carta, papeis, pulp, board, cardboard, paperboard | packaging, office, raw, material
wood = timber, lumber, madera, holz, bois, legno, plywood | raw, material, construction, packaging
cement = concrete, cemento, beton, aggregate, gravel, sand, stone, ballast, crushed, asphalt, bitumen, brick, block | construction, material, road, civil
# ---------------- packaging
pallet = pallets, palette, palet, tarima, tarimas, skid, skids, paleta, pallete, bancale | packaging, load, unit, logistic
packaging = package, packing, pack, embalaje, empaque, verpackung, emballage, imballaggio, embalagem, carton, cartons, box, boxes, caja, karton, carton, crate, bag, bags, sack, container | packaging, material
film = films, stretch, shrink, wrap, wrapping, foil, pellicola, pelicula, folie | packaging, wrapping, material
strapping = strap, straps, strapping, fleje, umreifung, feuillard, reggetta, banding | packaging, wrapping, material
label = labels, labelling, etiqueta, etikett, etiquette, sticker, tag | packaging, print, material
separator = separators, layer, pad, pads, slip, sheet, sheets, interlayer, divider, tier, frame, frames, topframe | packaging, pallet, load, unit
can = cans, lata, dose, canette, lattina, bottle, bottles, jar, closure, closures, cap, caps, lid, lids, end, ends | packaging, container, direct, material
# ---------------- logistics
freight = freights, frt, frght, fracht, flete, fletes, frete, fret, nolo, cargo, haulage, carriage, trucking, truck, lorry, shipping, shipment, ship, sea, ocean, vessel, container, ltl, ftl, transport, transportation, transporte, trasporto, logistics, logistica, forwarding, forwarder, delivery | freight, logistics, transport, road, sea
courier = couriers, express, parcel, parcels, dhl, fedex, ups, tnt, postage, postal, mail, stamps, correo, porto, paqueteria, mensajeria, envio | courier, postage, freight, logistics
customs = custom, aduana, aduanas, zoll, douane, dogana, alfandega, duty, duties, import, export, broker, brokerage, clearance, tariff, despacho | customs, logistics, clearance, duty
warehouse = warehousing, storage, almacen, lager, entrepot, magazzino, armazem, 3pl, fulfilment, fulfillment | logistics, storage, facility
# ---------------- utilities / energy
electricity = electricity, power, kwh, energia, energie, electricidad, strom, electricite, elettricita, grid, utility, utilities, eskom | energy, utility, electricity
gas = gases, natural, lpg, propane, propano, butane, cng, gaz, gas, erdgas, methane | energy, utility, fuel
fuel = fuels, diesel, petrol, gasoline, gasolina, kraftstoff, carburant, carburante, combustivel, combustible, kerosene, paraffin, jet, avgas, adblue | energy, fuel, fleet, vehicle
water = agua, wasser, eau, acqua, agua, sewer, sewage, effluent, wastewater, alcantarillado | utility, water
# ---------------- facilities
cleaning = clean, cleaner, janitorial, janitor, housekeeping, limpieza, reinigung, nettoyage, pulizia, limpeza, hygiene, sanitation, washroom, pest, fumigation, laundry | facility, service, soft
security = guard, guards, guarding, seguridad, sicherheit, securite, sicurezza, surveillance, cctv, alarm, patrol, access, vigilancia | facility, service
waste = wastes, garbage, trash, refuse, rubbish, basura, residuos, abfall, dechets, rifiuti, residuo, disposal, recycling, recycle, reciclaje, ewaste, skip, landfill, scrap | environmental, facility, service, waste
rental = rent, rentals, hire, renta, alquiler, arriendo, miete, location, noleggio, aluguel, lease, leasing | rental, equipment, facility, site
maintenance = maint, mntn, mnt, mantenimiento, manutencion, wartung, instandhaltung, entretien, manutenzione, manutencao, servicing, upkeep, overhaul, preventive, corrective | maintenance, repair, service
repair = repairs, repr, rpr, reparacion, reparatur, reparation, riparazione, reparo, fix, refurbishment, refurb, rebuild, recondition, reconditioning | repair, maintenance, service
building = buildings, build, construction, civil, renovation, refurbishment, facility, facilities, premises, office, plant, site, hvac, plumbing, roofing, painting, carpentry, electrician | facility, maintenance, construction
canteen = cafeteria, catering, food, meal, meals, lunch, comedor, kantine, cantine, mensa, refectory, vending, coffee, beverage, beverages, kitchen, restaurant | canteen, catering, employee, hospitality
office = offices, stationery, papeleria, buero, bureau, ufficio, escritorio, pen, pens, toner, printer, copier, envelope, folder, binder, notebook | office, supply, stationery
furniture = furnitures, mueble, muebles, moebel, meuble, mobili, desk, chair, chairs, cabinet, shelving, locker | office, facility
# ---------------- IT and telecom
hardware = hw, computer, computers, laptop, laptops, notebook, desktop, pc, server, servers, monitor, screen, keyboard, mouse, dock, docking, tablet, printer, switch, router, firewall, network, networking, storage, nas, san, dell, hp, lenovo, cisco, apple | it, hardware, end-user, network
software = sw, licence, license, licences, licenses, licensing, subscription, subscriptions, saas, cloud, microsoft, oracle, sap, adobe, salesforce, renewal, renew, app, application, platform, hosting, azure, aws | it, software, licence, subscription
it = ict, informatica, informatique, tecnologia, technology, digital, data, helpdesk, support, hosting | it, technology
telecom = telecommunication, telecommunications, telephone, telephony, phone, phones, mobile, cellular, cell, voice, data, internet, broadband, bandwidth, sim, airtime, telefono, telefonia, telefon, wan, lan, mpls, landline | telecom, it, voice, data
# ---------------- professional services
consulting = consultant, consultants, consultancy, consultoria, beratung, conseil, consulenza, advisory, advisor, professional, sow, project, implementation, contractor, contractors, outsourcing | professional, service, consulting
legal = law, lawyer, lawyers, attorney, attorneys, solicitor, notary, litigation, abogado, abogados, rechtsanwalt, avocat, avvocato, juridico, fees | legal, professional, advisory
audit = auditor, auditing, accounting, accountant, tax, bookkeeping, auditoria, wirtschaftspruefer, revision, actuarial, valuation | audit, advisory, professional, finance
labour = labor, temp, temporary, staffing, agency, manpower, contingent, recruitment, recruiting, headhunter, personal, personnel, arbeitnehmerueberlassung, interim, secondment, contract labour | labour, professional, hr, service
training = train, course, courses, certification, seminar, workshop, coaching, education, learning, capacitacion, formacion, schulung, formation, formazione, treinamento, tuition | training, hr, employee
marketing = advertising, advert, advertisement, media, campaign, promotion, promotional, branding, publicidad, werbung, publicite, pubblicita, sponsorship, events, event, exhibition, trade show, pr, public relations, agency, creative, design | marketing, service
printing = print, printed, imprenta, druck, impression, stampa, brochure, leaflet, signage, sign | marketing, print, office
insurance = insurances, seguro, seguros, versicherung, assurance, assicurazione, premium, broker, underwriting | insurance, finance, risk
bank = banking | finance
# ---------------- HR and employee
travel = travels, trvl, airfare, flight, flights, airline, ticket, hotel, hotels, accommodation, lodging, hotelski, smestaj, hospedaje, viaje, viajes, reise, voyage, viaggio, viagem, mileage, taxi, uber, car hire, per diem, subsistence, expense, expenses | travel, expense, employee
health = medical, medicine, medico, medicos, doctor, clinic, occupational, wellness, wellbeing, vaccination, physiotherapy, honorarios, examination, screening | health, occupational, employee, benefit
ppe = gloves, glove, guante, guantes, handschuh, gant, guanto, helmet, hardhat, goggles, glasses, respirator, mask, earplug, hearing, boots, boot, vest, coverall, overall, overalls, workwear, uniform, uniforms, apron, harness, lanyard, safety, protective, protector, firstaid, bandage, plaster | ppe, safety, first, aid
fire = extinguisher, extinguishers, extintor, sprinkler, sprinklers, feuerloescher, extincteur, estintore, hydrant, suppression, detection | fire, protection, safety
# ---------------- vehicles and equipment
vehicle = vehicles, car, cars, truck, trucks, van, bus, fleet, forklift, fork, lift, loader, tractor, trailer, dumper, lhd, auto, automotive, tyre, tire, tires, tyres, wheel, wheels, rim | vehicle, fleet, equipment
equipment = equip, eq, eqp, machine, machinery, machines, apparatus, unit, plant, maquinaria, maschine, machine, macchina | equipment, capital, machinery
engine = engines, motor, diesel, turbo, turbocharger, starter, alternator, radiator | engine, vehicle, equipment, spare
crane = cranes, hoist, hoists, winch, lifting, sling, slings, shackle, rope, wire rope, chain block | lifting, equipment, hoist, safety
compressor = compressors, compr, kompressor, compresseur, compressore, compresor | equipment, pneumatic, mechanical
conveyor = conveyors, cnvy, cnvyr, conv, conveyance, belt, transportador, foerderband, convoyeur, nastro | conveyor, equipment, mechanical
ventilation = vent, vnt, ventilacion, lueftung, fan, fans, blower, airflow, duct, ducting, ventilator | ventilation, equipment, fan
# ---------------- mining and heavy industry (general vocabulary)
explosive = explosives, expl, dynamite, emulsion, detonator, detonators, dtntr, blasting, blast, explosivo, sprengstoff, booster, fuse, igniter | explosive, mining, blasting
drill = drills, drilling, dril, drl, auger | drill, mining, tool
bit = bits, drillbit, button | drill, tool
rod = rods, bar, bars | steel, metal
anchor = anchors, anchr, rockbolt, rebar, rockanchor | bolt, roof, support, fastener
structure = structures, strct, wall, walls, stopping, stoppings, partition, brattice, curtain | construction, building, civil, ventilation
hire = hiring, hired | rental, labour, service
roof = roofing, rof | roof, support, mining
mesh = meshes, wld, welded, screen, grid, malla | mesh, support, wire
crusher = crushers, crushing, breaker, breakers, feeder, mill, grinding | crusher, equipment, mining
dust = stonedust, stnedst, powder, polvo, staub, poussiere | dust, stone, mining
seal = sealing, seals, stopping, stoppings, plug, bulkhead | seal, mining, ventilation
underground = ug, subterraneo, untertage | underground, mining
# ---------------- generic service words
service = services, svc, serv, srv, sv, servicio, servicios, dienstleistung, prestation, servizio, servico, labour, labor, work, works, callout, call | service
installation = install, inst, instl, instalacion, montage, installazione, commissioning, erection, assembly | installation, service, project
inspection = inspect, testing, test, calibration, calibrate, certification, survey, sampling, sample, analysis, laboratory, lab | inspection, testing, service, quality
supply = supplies, suministro, lieferung, fourniture, fornitura, fornecimento, provision | supply, material
material = materials, mat, materiales, materialien, materiel, materiale | material
"""

ABBREVIATIONS = {
    # common ERP / material-master abbreviations (industry independent)
    "assy": "assembly", "asmbly": "assembly", "brg": "bearing", "brng": "bearing", "vlv": "valve", "pmp": "pump",
    "mtr": "motor", "mntn": "maintenance", "maint": "maintenance", "mnt": "maintenance", "repr": "repair",
    "rpr": "repair", "elec": "electrical", "elect": "electrical", "mech": "mechanical", "hyd": "hydraulic",
    "hydr": "hydraulic", "pneu": "pneumatic", "pneum": "pneumatic", "fltr": "filter", "flt": "filter",
    "lub": "lubricant", "lube": "lubricant", "cbl": "cable", "cnvy": "conveyor", "cnvyr": "conveyor",
    "svc": "service", "serv": "service", "srv": "service", "sv": "service", "eqp": "equipment", "equip": "equipment",
    "eq": "equipment", "inst": "installation", "instl": "installation", "sprt": "support", "supp": "support",
    "blt": "bolt", "dril": "drill", "drl": "drill", "rck": "rock", "stl": "steel", "ug": "underground",
    "vnt": "ventilation", "vent": "ventilation", "xng": "crossing", "wld": "welded", "msh": "mesh",
    "swgr": "switchgear", "trnsp": "transport", "transp": "transport", "haz": "hazardous", "expl": "explosive",
    "dtntr": "detonator", "dtnr": "detonator", "compr": "compressor", "frt": "freight", "frght": "freight",
    "lic": "licence", "mgmt": "management", "sw": "software", "hw": "hardware", "trvl": "travel",
    "accom": "accommodation", "clng": "cleaning", "pwr": "power", "chem": "chemical", "lab": "laboratory",
    "med": "medical", "ofc": "office", "stnedst": "stonedust", "stndst": "stonedust", "ppe": "ppe",
    "rof": "roof", "rofblt": "roof bolt", "anchr": "anchor", "rubr": "rubber", "flng": "flange", "flw": "flow",
    "dia": "diameter", "lg": "length", "wd": "width", "thk": "thickness", "qty": "quantity", "pcs": "pieces",
    "ea": "each", "ctn": "carton", "plt": "pallet", "tech": "technician", "technl": "technical", "eng": "engineering",
    "engr": "engineering", "nt": "night", "labr": "labour", "brnch": "branch", "tyre": "tyre", "veh": "vehicle",
    "vehicle": "vehicle", "sec": "section", "prod": "production", "mnt": "maintenance", "insp": "inspection",
    "calib": "calibration", "consult": "consulting", "subs": "subscription", "ins": "insurance",
    "maintrepair": "maintenance repair", "maintrepairsv": "maintenance repair service",
    "flmprf": "flameproof", "fdr": "feeder", "brkr": "breaker", "bld": "build", "wal": "wall",
    "sngl": "single", "shft": "shift", "bltr": "bolter", "techn": "technician", "supvr": "supervisor",
    "mngr": "manager", "hlpr": "helper", "contr": "contractor", "frman": "foreman", "comp": "component",
}

STOPWORDS = {
    "the", "and", "for", "with", "of", "to", "in", "on", "at", "by", "from", "a", "an", "or", "per", "as", "is",
    "de", "del", "la", "el", "los", "las", "y", "en", "para", "con", "der", "die", "das", "und", "fur", "mit",
    "le", "les", "des", "du", "et", "pour", "avec", "il", "di", "e", "da", "per", "do", "dos", "das", "com",
    "x", "mm", "cm", "m", "kg", "g", "l", "ml", "ltr", "in", "inch", "ft", "lb", "lbs", "oz", "ea", "each", "pc",
    "pcs", "no", "nr", "qty", "set", "lot", "unit", "units", "item", "items", "type", "size", "sz", "p", "n",
    "pn", "ref", "see", "spec", "blanket", "po", "order", "invoice", "inv", "total", "misc", "various",
    "company", "co", "ltd", "pty", "inc", "llc", "gmbh", "sa", "cc", "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec", "month", "monthly", "annual", "year", "q1", "q2", "q3", "q4",
    "other", "others", "general", "id", "od", "std", "standard", "new", "old", "only",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@lru_cache(maxsize=200000)
def stem(w: str) -> str:
    """Tiny, language-light stemmer (plural/verb suffixes) so 'bearings'~'bearing', 'repairs'~'repair'."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(ch|sh|x|ss|zz)es$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _parse():
    syn: dict[str, set[str]] = {}       # stem -> set of synonym stems (incl. itself)
    rel: dict[str, set[str]] = {}       # stem -> set of broader stems
    for line in _RAW.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        head, rest = line.split("=", 1)
        syns, _, broader = rest.partition("|")
        words = [head.strip()] + [w.strip() for w in syns.split(",")]
        words = [strip_accents(w).lower() for w in words if w.strip()]
        group = set()
        for w in words:
            parts = [t for t in re.split(r"[^a-z0-9]+", w) if t]
            if len(parts) != 1:      # multi-word phrases would leak their parts ('car hire' -> 'car')
                continue
            if parts[0] not in STOPWORDS:
                group.add(stem(parts[0]))
        bset = set()
        for w in broader.split(","):
            for t in re.split(r"[^a-z0-9]+", strip_accents(w).lower()):
                if t and t not in STOPWORDS:
                    bset.add(stem(t))
        for g in group:
            syn.setdefault(g, set()).update(group)
            rel.setdefault(g, set()).update(bset - group)
    return syn, rel


SYNONYMS, BROADER = _parse()
ABBR = {k: [stem(t) for t in v.split()] for k, v in ABBREVIATIONS.items()}
