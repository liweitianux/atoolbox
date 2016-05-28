#include <cassert>

double ddivid(double (*foo)(double),double x1,double x2,double err)
{
  assert(x2>x1);
  assert(foo(x2)*foo(x1)<0);
  while(x2-x1>err)
    {
      double x=(x2+x1)/2.;
      double y=foo(x);
      double y1=foo(x1);
      double y2=foo(x2);
      if(y1*y<0)
	{
	  x2=x;
	}
      else if(y2*y<0)
	{
	  x1=x;
	}
      else
	{
	  assert(0);
	}

    }
  return (x1+x2)/2.;
}


double ddivid(double (*foo)(double),double z,double x1,double x2,double err)
{
  assert(x2>x1);
  assert((foo(x2)-z)*(foo(x1)-z)<0);
  while(x2-x1>err)
    {
      double x=(x2+x1)/2.;
      double y=foo(x)-z;
      double y1=foo(x1)-z;
      double y2=foo(x2)-z;
      if(y1*y<0)
	{
	  x2=x;
	}
      else if(y2*y<0)
	{
	  x1=x;
	}
      else
	{
	  assert(0);
	}

    }
  return (x1+x2)/2.;
}
