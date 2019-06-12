gen = MockStreamGenerator(df=FardalStreamDF(), hamiltonian=H)
stream = gen.run(prog_w0, dt=-1., n_steps=1000)


gen = MockStreamGenerator(df=FardalStreamDF(),
                          hamiltonian=H,
                          progenitor_potential=gp.PlummerPotential(),
                          progenitor_mass_loss=1*u.Msun/u.Myr)

stream = gen.run(prog_w0, dt=-1., n_steps=1000)


gen = MockStreamGenerator(df=FardalStreamDF(),
                          hamiltonian=H,
                          progenitor_potential=gp.PlummerPotential(),
                          progenitor_mass_loss=1*u.Msun/u.Myr)

stream = gen.run(prog_w0, nbody=nbody, dt=-1., n_steps=1000)
