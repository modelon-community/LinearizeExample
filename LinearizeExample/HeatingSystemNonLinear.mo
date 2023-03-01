within LinearizeExample;

model HeatingSystemNonLinear "Nonlinear heating system model"
  extends .LinearizeExample.Template.HeatingSystem;

  parameter Integer nu=2;
  parameter Integer ny=2;
  parameter Real u0[nu]={300,0.1};

  .Modelica.Blocks.Interfaces.RealInput u[nu](start=u0) annotation (Placement(transformation(
        extent={{-149.958238306213,-21.633744194200812},{-109.95823830621299,18.366255805799188}},
        origin={0.0,0.0},
        rotation=0.0)));
  .Modelica.Blocks.Interfaces.RealOutput y[ny] annotation (Placement(transformation(
        extent={{131.3317668008713,-11.123810602126806},{151.3317668008713,8.876189397873194}},
        origin={0.0,0.0},
        rotation=0.0)));

equation
  connect(sensor_T_return.T, y[2]) annotation (Line(points={{-37,-50},{-45.1642,-50},{-45.1642,
          3.87619},{141.332,3.87619}},
                              color={0,0,127}));
  connect(sensor_T_forward.T, y[1]) annotation (Line(points={{67,40},{117.584,40},{117.584,-6.12381},
          {141.332,-6.12381}}, color={0,0,127}));
  connect(u[1], burner.Q_flow) annotation (Line(points={{-129.958,-11.6337},{-93.1665,-11.6337},{
          -93.1665,57.2331},{-38.9639,57.2331},{-38.9639,40},{20,40}},
                                                              color={0,0,127}));
  connect(u[2], valve.opening) annotation (Line(points={{-129.958,8.36626},{-93.1665,8.36626},{
          -93.1665,-108.863},{50,-108.863},{50,-62}},
                                             color={0,0,127}));
  annotation (experiment(
      StopTime=10000,
      Tolerance=1e-006,
      __Dymola_Algorithm="Radau"), Documentation(info="<html><p><em>Modelica.Fluid.Examples.HeatingSystem</em>&nbsp;with input/output interfaces to analyze&nbsp;the impact of the burner heat rate and  valve opening on the forward 
and return temperatures. The modification are required to make&nbsp;linearization of that dynamics possible.</p></html>"));
end HeatingSystemNonLinear;
