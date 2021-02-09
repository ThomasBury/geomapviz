import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import holoviews
from src.geomapviz import prepare_geo_data, single_predictor_aggr_multi_mod, \
    plot_on_map, facet_map, facet_map_interactive


def _generate_dummy_data():
    # the greatest country in the world,
    # first military and economic power in the Universe
    shp_path = "../beshp/belgium.shp"
    geom_df = gpd.read_file(shp_path)

    # create correlation with the geo entities
    feat_1 = np.repeat(np.log10(geom_df.INS.astype(int).values), 10)
    feat_1 = (feat_1 - feat_1.min()) / (feat_1.max() - feat_1.min())
    # dummy data
    bel_df = pd.DataFrame({
        'geoid': np.repeat(geom_df.INS.values, 10),
        'truth': feat_1,
        'feat_2': feat_1 + np.random.beta(.5, .5, size=len(feat_1)),
        'feat_3': feat_1 + np.random.beta(2, 5, size=len(feat_1)),
        'feat_4': feat_1 + np.random.beta(5, 2, size=len(feat_1))
    }
    )
    return bel_df


class TestPrepareGeoData:

    def test_prepare_geo_data(self):
        bel_df = _generate_dummy_data()
        shp_path = "../beshp/belgium.shp"

        geo_df = prepare_geo_data(df=bel_df, target='truth', cols_to_plot=None, predicted=None, dissolve_on=None,
                                  distrib='gamma', n_bins=7, geoid='INS', weight=None, shp_path=shp_path)

        message = "The output df is not the right shape, the shape is {}, expected (589, 19)".format(geo_df.shape)
        assert geo_df.shape == (589, 19), message

    def test_prepare_geo_data_dissolve(self):
        bel_df = _generate_dummy_data()
        shp_path = "../beshp/belgium.shp"

        geo_df = prepare_geo_data(df=bel_df, target='truth', cols_to_plot=None, predicted=None, dissolve_on=None,
                                  distrib='gamma', n_bins=7, geoid='INS', weight=None, shp_path=shp_path)

        bel_df_borough = bel_df.merge(geo_df[['geoid', 'borough']])

        geo_df = prepare_geo_data(df=bel_df_borough, target='truth', cols_to_plot=None, predicted=None,
                                  dissolve_on='borough', distrib='gamma', n_bins=7, geoid='INS',
                                  weight=None, shp_path=shp_path)

        message = "The output df is not the right shape, the shape is {}, expected (43, 17)".format(geo_df.shape)
        assert geo_df.shape == (43, 17), message


class TestAggregator:

    def test_single_predictor_aggr_multi_mod(self):
        bel_df = _generate_dummy_data()
        df_average, df_ = single_predictor_aggr_multi_mod(df=bel_df, feature='feat_2', target='truth',
                                                          weight=None, predicted=None, distrib='gaussian', ci=True,
                                                          autobin=False,
                                                          n_vals=50, n_bins=20, names=None, verbose=False)

        message = "The output df is not the right shape, the shape is {}, expected (5890, 6)".format(df_average.shape)
        assert df_average.shape == (5890, 6), message

        message = "The output df is not the right shape, the shape is {}, expected (5890, 5)".format(df_.shape)
        assert df_.shape == (5890, 5), message


class TestPlotOnMap:

    def test_plot_on_map(self):
        bel_df = _generate_dummy_data()
        shp_path = "../beshp/belgium.shp"
        f = plot_on_map(df=bel_df, target='truth', dissolve_on=None, distrib='gaussian', plot_uncertainty=False,
                        plot_weight=False,
                        autobin=False, n_bins=7, geoid='INS', weight=None, shp_path=shp_path,
                        figsize=(20, 6), cmap=None, normalize=True, facecolor="black")

        message = "not the right type, got {} but expected matplotlib.figure.Figure".format(str(type(f)))
        assert isinstance(f, matplotlib.figure.Figure), message


class TestFacetMap:

    def test_plot_on_map(self):
        bel_df = _generate_dummy_data()
        shp_path = "../beshp/belgium.shp"
        f = facet_map(df=bel_df, target='truth', cols_to_plot=['feat_2', 'feat_3'],
                      predicted=None, dissolve_on=None,
                      autobin=True, n_bins=7, geoid='INS', weight=None, shp_path=shp_path,
                      figsize=(12, 12), ncols=2, cmap=None, normalize=False)

        message = "not the right type, got {} but expected matplotlib.figure.Figure".format(str(type(f)))
        assert isinstance(f, matplotlib.figure.Figure), message


class TestFacetMapInteractive:

    def test_facet_map_interactive(self):
        bel_df = _generate_dummy_data()
        shp_path = "../beshp/belgium.shp"
        f = facet_map_interactive(df=bel_df, target='truth', cols_to_plot=None, predicted=None, dissolve_on=None,
                                  autobin=True, n_bins=7, geoid='INS', weight=None, shp_path=shp_path,
                                  figsize=(400, 400), ncols=2, cmap=None, normalize=False)
        message = "not the right type, got {} but expected holoviews.core.layout.Layout".format(str(type(f)))
        isinstance(f, holoviews.core.layout.Layout), message
